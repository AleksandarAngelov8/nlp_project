#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Run:
#   python binoculars_m4_score.py ^
#     --data_root "C:\Users\kucem\PraktikumProject\nlp_project\repos\M4\data" ^
#     --include arxiv_ chatgpt davinci cohere flant5 dolly ^
#     --english_only --min_chars 40 --limit_per_class 50000 ^
#     --small distilgpt2 --large gpt2-medium --max_ctx 256 ^
#     --out "C:\Users\kucem\PraktikumProject\outputs\binoculars\m4_binoculars.csv"

import os, sys, csv, time, math, argparse, json, glob, hashlib, random
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Folder with M4 *.jsonl")
    ap.add_argument("--include", nargs="*", default=None,
                    help="Optional filename substrings to keep (e.g., arxiv_ chatgpt davinci cohere flant5 dolly)")
    ap.add_argument("--english_only", action="store_true", help="Heuristic ASCII-heavy filter")
    ap.add_argument("--min_chars", type=int, default=80, help="Drop texts shorter than this")
    ap.add_argument("--limit_per_class", type=int, default=50000, help="Cap per-class after filters; <=0 = no cap")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--small", default="distilgpt2", help="Small LM (for Binoculars)")
    ap.add_argument("--large", default="gpt2-medium", help="Large LM (for Binoculars)")
    ap.add_argument("--max_ctx", type=int, default=256, help="Max tokens for scoring")
    ap.add_argument("--batch", type=int, default=1, help="Keep 1 for low VRAM (per-row scoring)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--checkpoint_every", type=int, default=100, help="Flush every N rows")

    return ap.parse_args()

# -------------------- Utils --------------------
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def is_english_text(s: str) -> bool:
    if not isinstance(s, str):
        return False
    sample = s[:1000]
    non_ascii = sum(1 for ch in sample if ord(ch) > 127)
    return non_ascii <= len(sample) * 0.05

def discover_jsonl_files(root: str, include: Optional[List[str]]) -> List[str]:
    files = sorted(glob.glob(os.path.join(root, "*.jsonl")))
    if include:
        inc = [p.lower() for p in include]
        files = [f for f in files if any(p in os.path.basename(f).lower() for p in inc)]
    return files

# ---- robust coercion (handles str/list/dict + alt keys) ----
def _coerce_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return " ".join(_coerce_text(y) for y in x if y is not None)
    if isinstance(x, dict):
        for k in ("text", "content", "answer", "completion", "response", "abstract", "body"):
            if k in x:
                return _coerce_text(x[k])
        try:
            return " ".join(_coerce_text(v) for v in x.values())
        except Exception:
            return ""
    return str(x)

def _first_nonempty(obj: dict, keys: List[str]) -> str:
    for k in keys:
        if k in obj:
            s = _coerce_text(obj[k]).strip()
            if s:
                return s
    return ""

def load_m4_pairs(files: List[str], english_only: bool, min_chars: int) -> List[Dict]:
    """
    Emit records:
      {"text": <str>, "label": 0/1, "src":"M4", "kind":"human"/"ai", "path":<file>}
    """
    recs = []
    human_keys   = ["human_text", "abstract", "reference", "source_text", "human", "gold"]
    machine_keys = ["machine_text", "machine_abstract", "generated_text", "model_output", "prediction", "answer"]

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    continue

                h = _first_nonempty(obj, human_keys)
                m = _first_nonempty(obj, machine_keys)

                if not h:
                    h = _coerce_text(obj.get("prompt", "")).strip()
                if not m:
                    m = _coerce_text(obj.get("completion", obj.get("response", ""))).strip()

                if min_chars:
                    if len(h) < min_chars: h = ""
                    if len(m) < min_chars: m = ""
                if english_only:
                    if h and not is_english_text(h): h = ""
                    if m and not is_english_text(m): m = ""

                if h:
                    recs.append({"text": h, "label": 0, "src": "M4", "kind": "human", "path": fp})
                if m:
                    recs.append({"text": m, "label": 1, "src": "M4", "kind": "ai", "path": fp})
    return recs

# ----------------- Core scoring -----------------
def _ensure_min_tokens(input_ids: torch.Tensor,
                       attn: torch.Tensor,
                       tok) -> tuple[torch.Tensor, torch.Tensor]:
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    if eos_id is None:
        eos_id = 0
    T = input_ids.size(1)
    if T == 0:
        device = input_ids.device
        input_ids = torch.tensor([[eos_id, eos_id]], device=device, dtype=input_ids.dtype)
        attn = torch.tensor([[1, 1]], device=device, dtype=attn.dtype)
    elif T == 1:
        device = input_ids.device
        input_ids = torch.cat([input_ids, torch.tensor([[eos_id]], device=device, dtype=input_ids.dtype)], dim=1)
        attn = torch.cat([attn, torch.ones((1,1), device=device, dtype=attn.dtype)], dim=1)
    return input_ids, attn

def avg_nll(model, tok, texts, device, use_amp, max_ctx: int):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    enc = tok(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_ctx,
    )
    input_ids = enc["input_ids"].to(device, non_blocking=True)
    attn = enc["attention_mask"].to(device, non_blocking=True)

    input_ids, attn = _ensure_min_tokens(input_ids, attn, tok)

    with torch.inference_mode():
        if use_amp and device == "cuda":
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
    args = parse_args()
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    print(f"Device: {device}")

    # Tokenizers
    tok_s = AutoTokenizer.from_pretrained(args.small, use_fast=True)
    tok_l = AutoTokenizer.from_pretrained(args.large, use_fast=True)
    if tok_s.pad_token is None: tok_s.pad_token = tok_s.eos_token
    if tok_l.pad_token is None: tok_l.pad_token = tok_l.eos_token

    # Models
    if device == "cuda":
        mdl_s = AutoModelForCausalLM.from_pretrained(
            args.small, torch_dtype=torch.float16, device_map="auto"
        ).eval()
        mdl_l = AutoModelForCausalLM.from_pretrained(
            args.large, torch_dtype=torch.float16, device_map="auto"
        ).eval()
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        mdl_s = AutoModelForCausalLM.from_pretrained(args.small).eval()
        mdl_l = AutoModelForCausalLM.from_pretrained(args.large).eval()

    # Load M4
    files = discover_jsonl_files(args.data_root, args.include)
    if not files:
        print("[ERROR] No .jsonl files found with current --data_root/--include settings.")
        sys.exit(1)
    print(f"[info] Files considered: {len(files)}")

    recs = load_m4_pairs(files, english_only=args.english_only, min_chars=args.min_chars)
    print(f"[info] Records after filters: {len(recs)}")
    human = [r for r in recs if r["label"] == 0]
    ai    = [r for r in recs if r["label"] == 1]
    random.shuffle(human); random.shuffle(ai)

    if args.limit_per_class and args.limit_per_class > 0:
        human = human[:args.limit_per_class]
        ai    = ai[:args.limit_per_class]

    rows = human + ai
    random.shuffle(rows)
    print(f"[info] Using {len(rows)} rows (balanced={min(len(human), len(ai))} per class).")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Resume via text_hash
    done_hashes = set()
    if os.path.exists(args.out):
        print(f"[INFO] Resuming from existing file {args.out}")
        try:
            df_done = pd.read_csv(args.out)
            if "text_hash" in df_done.columns:
                done_hashes = set(df_done["text_hash"].astype(str).tolist())
            else:
                # fallback for very old runs: try 'text' if present
                if "text" in df_done.columns:
                    done_hashes = set(df_done["text"].astype(str).map(lambda s: hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()))
        except Exception as e:
            print(f"[WARN] Could not read existing CSV for resume: {e}")

    # Prepare CSV (header if new)
    if not os.path.exists(args.out):
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # keep it consistent with your other CSVs, avoid dumping full text
            w.writerow(["key","path","text_hash","label","nll_small","nll_large","score","pred","src","kind"])

    # Count how many NEW rows remain
    to_process = 0
    for r in rows:
        th = sha1(r["text"])
        if th not in done_hashes:
            to_process += 1
    if to_process == 0:
        print("[INFO] Nothing to do (all rows already processed).")
        return

    # ETA state
    processed_new = 0
    avg_row_time = None   # EMA seconds/row
    ema_alpha = 0.2
    start_wall = datetime.now()

    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        pbar = tqdm(total=to_process, desc="Scoring", unit="row")

        for r in rows:
            txt = r["text"]
            th  = sha1(txt)
            if th in done_hashes:
                continue

            row_t0 = time.perf_counter()
            try:
                nll_s = avg_nll(mdl_s, tok_s, [txt], device, use_amp, args.max_ctx)[0]
                nll_l = avg_nll(mdl_l, tok_l, [txt], device, use_amp, args.max_ctx)[0]
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg:
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    print("[ERROR] CUDA OOM. Reduce --max_ctx or use smaller models.")
                    sys.exit(1)
                if "0 elements" in msg or "reshape" in msg:
                    print("[WARN] Skipping a problematic row due to zero-length encoding.")
                    continue
                raise

            score = nll_l - nll_s  # higher => more "AI-like"
            pred = int(score > 0)  # naive threshold

            key = sha1(f'{r["label"]}|{th}|{os.path.basename(r["path"])}')
            w.writerow([key, r["path"], th, r["label"],
                        f"{nll_s:.6f}", f"{nll_l:.6f}", f"{score:.6f}", pred, r["src"], r["kind"]])

            processed_new += 1

            # checkpointing
            if (processed_new % args.checkpoint_every) == 0:
                f.flush()
                os.fsync(f.fileno())
                print(f"[Checkpoint] {processed_new} / {to_process} processed")

            # ETA
            row_dt = time.perf_counter() - row_t0
            avg_row_time = row_dt if avg_row_time is None else (ema_alpha * row_dt + (1 - ema_alpha) * avg_row_time)
            left = to_process - processed_new
            eta_sec = left * (avg_row_time or 0.0)
            fin = datetime.now() + timedelta(seconds=eta_sec)
            rps = (1.0 / avg_row_time) if avg_row_time and avg_row_time > 0 else 0.0

            pbar.set_postfix({
                "r/s": f"{rps:.2f}",
                "avg_s/row": f"{(avg_row_time or 0):.3f}",
                "ETA": _format_eta(eta_sec),
                "fin": fin.strftime("%H:%M:%S")
            })
            pbar.update(1)

    total_elapsed = datetime.now() - start_wall
    print(f"[DONE] Wrote {args.out} | elapsed {str(total_elapsed).split('.')[0]}")

if __name__ == "__main__":
    main()
