# save as binoculars_score.py ; run: python binoculars_score.py
import os, sys, csv, time, math
from datetime import datetime, timedelta

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pandas as pd

# -------------------- Config --------------------
SMALL = "distilgpt2"
LARGE = "gpt2-medium"

MAX_CTX = 256           # reduce if OOM: 192 or 128
BATCH = 1               # per-row scoring; keep 1 for low VRAM
MAX_ROWS = None         # None = all, or int to limit

OUT = r"C:\Users\kucem\PraktikumProject\outputs\binoculars\hc3_binoculars.csv"
CHECKPOINT_EVERY = 100  # save progress every N scored rows

# ----------------- HC3 loader (fixes HC3.py script issue) -----------------
def _warn_if_local_hc3_script():
    suspicious = []
    for p in ["HC3.py", os.path.join(os.getcwd(), "HC3.py")]:
        if os.path.isfile(p):
            suspicious.append(os.path.abspath(p))
    for d in sys.path:
        try:
            fp = os.path.join(d, "HC3.py")
            if os.path.isfile(fp):
                suspicious.append(os.path.abspath(fp))
        except Exception:
            pass
    if suspicious:
        print("[WARN] Found local HC3.py file(s) that can break datasets:", *suspicious, sep="\n  ")

def load_hc3(subset_name: str = "all"):
    """
    Load HC3 from the Hub as raw JSON/JSONL (datasets v4+ compatible).
    Tries <subset>.jsonl then <subset>.json and returns a single 'train' split.
    """
    _warn_if_local_hc3_script()
    last_err = None
    for ext in ("jsonl", "json"):
        try:
            fp = hf_hub_download(
                repo_id="Hello-SimpleAI/HC3",
                filename=f"{subset_name}.{ext}",
                repo_type="dataset",
            )
            ds = load_dataset("json", data_files=fp, split="train")
            if len(ds) > 0:
                return ds
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Could not load HC3 subset '{subset_name}' as .jsonl/.json. Last error: {repr(last_err)}"
    )

def iter_rows(max_rows=None):
    ds = load_hc3("all")
    cnt = 0
    for ex in ds:
        # Skip rows with empty strings early
        for h in (ex.get("human_answers") or []):
            if h and h.strip():
                yield {"text": h, "label": 0, "src": "HC3", "kind": "human"}
                cnt += 1
                if max_rows and cnt >= max_rows: return
        for a in (ex.get("chatgpt_answers") or []):
            if a and a.strip():
                yield {"text": a, "label": 1, "src": "HC3", "kind": "ai"}
                cnt += 1
                if max_rows and cnt >= max_rows: return

# ----------------- Core scoring -----------------
def _ensure_min_tokens(input_ids: torch.Tensor,
                       attn: torch.Tensor,
                       tok) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ensure sequence length >= 2 so shifting ([:-1], [1:]) is valid.
    Works with BATCH=1 (shape [1, T]).
    """
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    if eos_id is None:
        # Fallback to an arbitrary token id 0 if tokenizer strangely lacks eos/pad
        eos_id = 0

    T = input_ids.size(1)
    if T == 0:
        # Create [eos, eos]
        device = input_ids.device
        input_ids = torch.tensor([[eos_id, eos_id]], device=device, dtype=input_ids.dtype)
        attn = torch.tensor([[1, 1]], device=device, dtype=attn.dtype)
    elif T == 1:
        # Append eos to make length 2
        device = input_ids.device
        input_ids = torch.cat([input_ids, torch.tensor([[eos_id]], device=device, dtype=input_ids.dtype)], dim=1)
        attn = torch.cat([attn, torch.ones((1,1), device=device, dtype=attn.dtype)], dim=1)
    return input_ids, attn

def avg_nll(model, tok, texts, device, use_amp):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    enc = tok(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_CTX,
    )
    input_ids = enc["input_ids"].to(device, non_blocking=True)
    attn = enc["attention_mask"].to(device, non_blocking=True)

    # Guarantee at least two tokens (handles empty/1-token cases)
    input_ids, attn = _ensure_min_tokens(input_ids, attn, tok)

    with torch.inference_mode():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_ids=input_ids, attention_mask=attn).logits
        else:
            logits = model(input_ids=input_ids, attention_mask=attn).logits

        logits = logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()
        mask = attn[:, 1:].contiguous()

        loss_tok = F.cross_entropy(
            logits.transpose(1, 2),  # [B, V, T-1]
            labels,
            reduction="none",
        )                            # [B, T-1]

        loss_tok = loss_tok * mask
        tok_counts = mask.sum(dim=1).clamp(min=1)
        nll = (loss_tok.sum(dim=1) / tok_counts)
        return nll.detach().cpu().tolist()

# ----------------- Helpers for ETA -----------------
def _format_eta(seconds: float) -> str:
    if seconds is None or math.isinf(seconds) or math.isnan(seconds) or seconds < 0:
        return "--:--:--"
    seconds = int(seconds + 0.5)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# ----------------- Main with checkpointing + ETA -----------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    print(f"Device: {device}")

    # Tokenizers
    tok_s = AutoTokenizer.from_pretrained(SMALL, use_fast=True)
    tok_l = AutoTokenizer.from_pretrained(LARGE, use_fast=True)
    if tok_s.pad_token is None: tok_s.pad_token = tok_s.eos_token
    if tok_l.pad_token is None: tok_l.pad_token = tok_l.eos_token

    # Models
    if device == "cuda":
        mdl_s = AutoModelForCausalLM.from_pretrained(
            SMALL, torch_dtype=torch.float16, device_map="auto"
        ).eval()
        mdl_l = AutoModelForCausalLM.from_pretrained(
            LARGE, torch_dtype=torch.float16, device_map="auto"
        ).eval()
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        mdl_s = AutoModelForCausalLM.from_pretrained(SMALL).eval()
        mdl_l = AutoModelForCausalLM.from_pretrained(LARGE).eval()

    # Load data
    rows = list(iter_rows(MAX_ROWS))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    # Resume if checkpoint exists
    done_keys = set()
    if os.path.exists(OUT):
        print(f"[INFO] Resuming from existing file {OUT}")
        df_done = pd.read_csv(OUT)
        done_keys = set(df_done["text"].tolist())

    # Count how many NEW rows remain
    total_to_process = sum(1 for r in rows if r["text"] not in done_keys)
    if total_to_process == 0:
        print("[INFO] Nothing to do (all rows already processed).")
        return

    # Prepare CSV (write header only if new file)
    if not os.path.exists(OUT):
        with open(OUT, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["text","label","nll_small","nll_large","score","pred","src","kind"])

    # ETA state
    processed_new = 0
    avg_row_time = None   # EMA of per-row seconds
    ema_alpha = 0.2       # smoothing factor
    start_wall = datetime.now()

    # Progress bar counts only *newly scored* rows
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        pbar = tqdm(total=total_to_process, desc="Scoring", unit="row")

        for r in rows:
            if r["text"] in done_keys:
                continue  # already done

            row_t0 = time.perf_counter()

            texts = [r["text"]]
            try:
                nll_s = avg_nll(mdl_s, tok_s, texts, device, use_amp)[0]
                nll_l = avg_nll(mdl_l, tok_l, texts, device, use_amp)[0]
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg:
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    print("[WARN] CUDA OOM, try reducing MAX_CTX or switching to smaller models.")
                    sys.exit(1)
                # Handle any odd empty/shape issues gracefully by skipping
                if "0 elements" in msg or "reshape" in msg:
                    print("[WARN] Skipping a problematic row due to zero-length encoding.")
                    continue
                raise

            score = nll_l - nll_s  # higher => "AI-like"
            pred = int(score > 0)  # naive threshold

            w.writerow([r["text"], r["label"], f"{nll_s:.6f}", f"{nll_l:.6f}",
                        f"{score:.6f}", pred, r["src"], r["kind"]])

            processed_new += 1

            # Flush occasionally for safety
            if (processed_new % CHECKPOINT_EVERY) == 0:
                f.flush()
                os.fsync(f.fileno())
                print(f"[Checkpoint] Processed {processed_new} / {total_to_process}")

            # --- Update ETA ---
            row_dt = time.perf_counter() - row_t0
            if avg_row_time is None:
                avg_row_time = row_dt
            else:
                avg_row_time = ema_alpha * row_dt + (1 - ema_alpha) * avg_row_time

            rows_left = total_to_process - processed_new
            eta_sec = rows_left * avg_row_time if avg_row_time and rows_left > 0 else 0.0
            finish_time = datetime.now() + timedelta(seconds=eta_sec)
            rps = (1.0 / avg_row_time) if avg_row_time and avg_row_time > 0 else 0.0

            pbar.set_postfix({
                "r/s": f"{rps:.2f}",
                "avg_s/row": f"{avg_row_time:.3f}",
                "ETA": _format_eta(eta_sec),
                "fin": finish_time.strftime("%H:%M:%S")
            })
            pbar.update(1)

    total_elapsed = datetime.now() - start_wall
    print(f"[DONE] Wrote {OUT} | elapsed {str(total_elapsed).split('.')[0]}")

if __name__ == "__main__":
    main()
