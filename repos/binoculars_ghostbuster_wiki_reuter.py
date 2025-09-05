# binoculars_ghostbuster_wiki_reuter.py
import os, sys, csv, glob, hashlib, random, argparse, math, time, warnings
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.logging import set_verbosity_error

# Silence HF’s chatty warnings and the long-sequence message
set_verbosity_error()
warnings.filterwarnings(
    "ignore",
    message="Token indices sequence length is longer than the specified maximum",
)

DEFAULT_SMALL = "distilgpt2"
DEFAULT_LARGE = "gpt2-medium"
DEFAULT_MAX_CTX = 256
DEFAULT_OVERLAP = 32
DEFAULT_BATCH = 1
CHECKPOINT_EVERY = 100

def read_text(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def _has_txt(base: str) -> bool:
    pats = [
        os.path.join(base, "human", "**", "*.txt"),
        os.path.join(base, "gpt", "**", "*.txt"),
        os.path.join(base, "gpt_prompt1", "**", "*.txt"),
        os.path.join(base, "gpt_prompt2", "**", "*.txt"),
        os.path.join(base, "gpt_semantic", "**", "*.txt"),
        os.path.join(base, "gpt_writing", "**", "*.txt"),
        os.path.join(base, "claude", "**", "*.txt"),
    ]
    for p in pats:
        if glob.glob(p, recursive=True):
            return True
    return False

def resolve_domain_base(root_dir: str, domain: str, data_root: str | None) -> str:
    candidates = []
    if data_root:
        if os.path.basename(os.path.normpath(data_root)).lower() == domain.lower():
            candidates.append(data_root)
        else:
            candidates.append(os.path.join(data_root, domain))
    # IMPORTANT: include ghostbuster/data/<domain> (your layout)
    candidates += [
        os.path.join(root_dir, "ghostbuster", "data", domain),
        os.path.join(root_dir, "data", "ghostbuster-data", domain),
        os.path.join(root_dir, "ghostbuster", "data", "ghostbuster-data", domain),
        os.path.join(root_dir, "ghostbuster-data", domain),
        os.path.join(root_dir, "data", domain),
        os.path.join(root_dir, domain),
    ]
    tried = []
    for c in candidates:
        tried.append(c)
        if os.path.isdir(c) and _has_txt(c):
            return c
    raise FileNotFoundError(
        "Could not find any *.txt files for domain '{d}'. Tried:\n  - "
        + "\n  - ".join(tried)
    )

def load_ghostbuster_paths(root_dir: str, domain: str, data_root: str | None):
    base = resolve_domain_base(root_dir, domain, data_root)
    human_paths = sorted(glob.glob(os.path.join(base, "human", "**", "*.txt"), recursive=True))
    ai_dirs = ["gpt", "gpt_prompt1", "gpt_prompt2", "gpt_semantic", "gpt_writing", "claude"]
    ai_paths = []
    for d in ai_dirs:
        ai_paths.extend(glob.glob(os.path.join(base, d, "**", "*.txt"), recursive=True))
    ai_paths = sorted(ai_paths)
    print("Using domain base:", base)
    print("#human txt:", len(human_paths), "| #ai txt:", len(ai_paths))
    if not human_paths or not ai_paths:
        raise FileNotFoundError(
            f"No files found for one or both classes under {base}.\n"
            f"human={len(human_paths)}, ai={len(ai_paths)}"
        )
    return human_paths, ai_paths

def balanced_sample(human_paths, ai_paths, n_per_class, seed=42):
    random.seed(seed)
    random.shuffle(human_paths)
    random.shuffle(ai_paths)
    n = min(n_per_class, len(human_paths), len(ai_paths))
    return human_paths[:n], ai_paths[:n]

def _safe_texts_for_tok(texts, tok):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos = tok.eos_token or " "
    out = []
    for t in texts:
        s = (t or "").strip()
        out.append(s if s else eos)
    return out

@torch.inference_mode()
def nll_one_sliding(text: str, tok, model, device, max_ctx: int, overlap: int) -> float:
    try:
        tok.model_max_length = int(1e9)  # avoid tokenizer length warnings
    except Exception:
        pass

    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) < 2:
        if tok.eos_token_id is not None:
            ids = ids + [tok.eos_token_id]
        else:
            ids = ids + [ids[-1] if ids else 0]

    step = max(1, max_ctx - overlap)
    total_loss, total_tokens = 0.0, 0

    for start in range(0, len(ids), step):
        chunk_ids = ids[start : start + max_ctx]
        if len(chunk_ids) < 2:
            break
        input_ids = torch.tensor(chunk_ids, dtype=torch.long, device=device).unsqueeze(0)
        attn = torch.ones_like(input_ids, device=device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        logits = logits[:, :-1, :]
        labels = input_ids[:, 1:]
        mask   = attn[:, 1:]
        loss_tok = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
        loss_tok = loss_tok * mask
        valid = int(mask.sum().item())
        if valid > 0:
            total_loss  += float(loss_tok.sum().item())
            total_tokens += valid
        if start + max_ctx >= len(ids):
            break

    if total_tokens == 0:
        return 0.0
    return total_loss / total_tokens

def avg_nll_sliding(model, tok, texts, device, max_ctx: int, overlap: int):
    texts = _safe_texts_for_tok(texts, tok)
    return [nll_one_sliding(t, tok, model, device, max_ctx, overlap) for t in texts]

def _fmt_eta(secs: float) -> str:
    if secs is None or math.isnan(secs) or math.isinf(secs) or secs < 0:
        return "--:--:--"
    secs = int(secs + 0.5)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=os.path.abspath("."), help="Project root")
    ap.add_argument("--data_root", type=str, default=None,
                    help="Path to folder containing essay/reuter/wp or directly the domain folder")
    ap.add_argument("--domain", type=str, required=True, choices=["essay","reuter","wp"])
    ap.add_argument("--n_per_class", type=int, default=1000)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max_ctx", type=int, default=DEFAULT_MAX_CTX)
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--small_model", type=str, default=DEFAULT_SMALL)
    ap.add_argument("--large_model", type=str, default=DEFAULT_LARGE)
    ap.add_argument("--large_on_cpu", action="store_true")
    ap.add_argument("--log_every", type=int, default=50, help="Print heartbeat every N files")
    args = ap.parse_args()

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    small_device = "cuda" if torch.cuda.is_available() else "cpu"
    large_device = "cpu" if args.large_on_cpu else small_device
    print(f"Small device: {small_device} | Large device: {large_device}")
    print(f"Models: small='{args.small_model}'  large='{args.large_model}'")
    print(f"Sliding windows: max_ctx={args.max_ctx}, overlap={args.overlap}")

    tok_s = AutoTokenizer.from_pretrained(args.small_model, use_fast=True)
    tok_l = AutoTokenizer.from_pretrained(args.large_model, use_fast=True)
    if tok_s.pad_token is None: tok_s.pad_token = tok_s.eos_token
    if tok_l.pad_token is None: tok_l.pad_token = tok_l.eos_token

    if small_device == "cuda":
        mdl_s = AutoModelForCausalLM.from_pretrained(
            args.small_model, dtype=torch.float16, device_map={"": small_device}
        ).eval()
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        mdl_s = AutoModelForCausalLM.from_pretrained(args.small_model).to(small_device).eval()

    mdl_l = AutoModelForCausalLM.from_pretrained(
        args.large_model, dtype=(torch.float16 if large_device == "cuda" else None)
    ).to(large_device).eval()

    human_paths, ai_paths = load_ghostbuster_paths(args.root, args.domain, args.data_root)
    h_sel, a_sel = balanced_sample(human_paths, ai_paths, args.n_per_class)
    rows = (
        [{"path": p, "label": 0, "src": f"GB-{args.domain}", "kind": "human"} for p in h_sel] +
        [{"path": p, "label": 1, "src": f"GB-{args.domain}", "kind": "ai"} for p in a_sel]
    )
    random.shuffle(rows)

    done_keys = set()
    if os.path.exists(out_path):
        print(f"[INFO] Resuming from {out_path}")
        df_done = pd.read_csv(out_path)
        if "text_hash" in df_done.columns:
            done_keys = set(df_done["text_hash"].astype(str).tolist())
        elif "path" in df_done.columns:
            done_keys = set(df_done["path"].astype(str).tolist())
    else:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path","text_hash","label","nll_small","nll_large","score","pred","src","kind"])

    start = time.perf_counter()
    ema_s_per_row = None
    ema_alpha = 0.3
    processed_new = 0
    wrote_since_flush = 0

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        pbar = tqdm(
            rows, desc=f"Binoculars on GB-{args.domain} (ctx={args.max_ctx}, overlap={args.overlap})",
            unit="file", miniters=1, mininterval=0.3, smoothing=0.1, dynamic_ncols=True
        )

        for idx, r in enumerate(pbar, 1):
            if r["path"] in done_keys:
                continue

            t0 = time.perf_counter()
            text = read_text(r["path"])
            thash = sha1(text)
            if thash in done_keys:
                continue

            try:
                a = nll_one_sliding(text, tok_s, mdl_s, next(mdl_s.parameters()).device, args.max_ctx, args.overlap)
                b = nll_one_sliding(text, tok_l, mdl_l, next(mdl_l.parameters()).device, args.max_ctx, args.overlap)
                score = a - b
                pred  = int(score > 0)
                w.writerow([r["path"], thash, r["label"],
                            f"{a:.6f}", f"{b:.6f}", f"{score:.6f}", pred, r["src"], r["kind"]])
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("\n[ERROR] CUDA OOM. Reduce --max_ctx (e.g., 192/128) and/or try --large_on_cpu.")
                    sys.exit(1)
                # write a row marking failure & continue
                w.writerow([r["path"], thash, r["label"], "", "", "", "", r["src"], r["kind"]])
            except Exception as e:
                # keep going on random decoding/model errors
                w.writerow([r["path"], thash, r["label"], "", "", "", "", r["src"], r["kind"]])

            wrote_since_flush += 1
            processed_new += 1
            dt = time.perf_counter() - t0
            s_per_row = dt
            ema_s_per_row = s_per_row if ema_s_per_row is None else ema_alpha * s_per_row + (1 - ema_alpha) * ema_s_per_row
            remained = max(len(rows) - processed_new, 0)
            eta_sec = remained * (ema_s_per_row or 0.0)
            pbar.set_postfix({
                "r/s": f"{(1.0/(ema_s_per_row or 1e-9)):.2f}",
                "ETA": _fmt_eta(eta_sec),
            })

            if wrote_since_flush >= CHECKPOINT_EVERY:
                f.flush(); os.fsync(f.fileno()); wrote_since_flush = 0

            # heartbeat
            if args.log_every > 0 and (processed_new % args.log_every == 0):
                print(f"[{processed_new}/{len(rows)}] ~{(1.0/(ema_s_per_row or 1e-9)):.2f} r/s | ETA {_fmt_eta(eta_sec)}")

        f.flush(); os.fsync(f.fileno())

    elapsed = time.perf_counter() - start
    print(f"[DONE] Wrote {out_path} | processed={processed_new} | elapsed {elapsed:.1f}s")

if __name__ == "__main__":
    main()
