# binoculars_ghostbuster_std.py
# Usage example:
#   python binoculars_ghostbuster_std.py --domain wp --data_root /path/to/ghostbuster-data --out out_wp.csv
#   python binoculars_ghostbuster_std.py --domain reuter --n_per_class 3000 --out out_reuter.csv
#
# Outputs unified CSV:
# id,dataset,domain,detector,y_true,y_score,pred,threshold,text_len,meta
# - dataset   = "ghostbuster"
# - domain    = {"gb-wp","gb-reuter","gb-essay"}
# - detector  = f"binoculars-open:{small_model}->${large_model}"
# - y_score   = sigmoid(nll_small - nll_large)  # higher ⇒ more AI
# - pred, threshold left blank (no fixed threshold)
# - meta      = JSON (path, text_hash, models, max_ctx, overlap)

import os, sys, csv, glob, hashlib, random, argparse, math, time, json
from typing import List, Tuple
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------- Defaults (4 GB VRAM friendly) --------------------
DEFAULT_SMALL = "distilgpt2"
DEFAULT_LARGE = "gpt2-medium"
MAX_CTX_DEFAULT = 256
OVERLAP_DEFAULT = 32
BATCH = 1
CHECKPOINT_EVERY = 100

# -------------------- Robust file reading --------------------
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

# -------------------- Domain base resolution --------------------
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
    candidates += [
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

# -------------------- Dataset loader --------------------
def load_ghostbuster_paths(root_dir: str, domain: str, data_root: str | None) -> Tuple[List[str], List[str]]:
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

def balanced_sample(human_paths: List[str], ai_paths: List[str], n_per_class: int, seed: int = 42,
                    enforce_exact: bool = True) -> Tuple[List[str], List[str]]:
    random.seed(seed)
    random.shuffle(human_paths)
    random.shuffle(ai_paths)
    if enforce_exact and (len(human_paths) < n_per_class or len(ai_paths) < n_per_class):
        raise ValueError(
            f"Requested n_per_class={n_per_class} but available human={len(human_paths)} ai={len(ai_paths)}."
        )
    n = min(n_per_class, len(human_paths), len(ai_paths))
    return human_paths[:n], ai_paths[:n]

# -------------------- Sliding-window NLL scorer --------------------
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
        tok.model_max_length = int(1e9)  # silence tokenizer warnings
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
            total_loss += float(loss_tok.sum().item())
            total_tokens += valid
        if start + max_ctx >= len(ids):
            break
    if total_tokens == 0:
        return 0.0
    return total_loss / total_tokens

def avg_nll_sliding(model, tok, texts, device, max_ctx: int, overlap: int):
    texts = _safe_texts_for_tok(texts, tok)
    return [nll_one_sliding(t, tok, model, device, max_ctx, overlap) for t in texts]

# -------------------- Utilities --------------------
def _fmt_eta(secs: float) -> str:
    if secs is None or math.isnan(secs) or math.isinf(secs) or secs < 0:
        return "--:--:--"
    secs = int(secs + 0.5)
    h, rem = divmod(secs, 3600); m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def sigmoid(x: float) -> float:
    # numerically stable for big |x|
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

# -------------------- Two-pass NLL computation (VRAM friendly) --------------------
def compute_nlls_for_model(model_id: str, texts: List[str], device: str,
                           max_ctx: int, overlap: int, desc: str) -> List[float]:
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if device == "cuda":
        mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda").eval()
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        mdl = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()

    nlls = []
    pbar = tqdm(range(0, len(texts), BATCH), desc=desc, unit="row")
    for i in pbar:
        batch = texts[i : i + BATCH]
        vals = avg_nll_sliding(mdl, tok, batch, next(mdl.parameters()).device, max_ctx, overlap)
        nlls.extend(vals)
    # cleanup
    del mdl; del tok
    if device == "cuda":
        torch.cuda.empty_cache()
    return nlls

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=os.path.abspath("."), help="Project root (parent of data/…)")
    ap.add_argument("--data_root", type=str, default=None, help="Path to folder containing essay/reuter/wp or the domain folder itself")
    ap.add_argument("--domain", type=str, required=True, choices=["essay","reuter","wp"], help="Ghostbuster domain")
    ap.add_argument("--n_per_class", type=int, default=3000, help="Balanced samples per class (3k ⇒ 6k rows)")
    ap.add_argument("--no_enforce_exact", action="store_true", help="If set, do not error when fewer than n_per_class are available")
    ap.add_argument("--out", type=str, required=True, help="Output CSV path (unified schema)")
    ap.add_argument("--max_ctx", type=int, default=MAX_CTX_DEFAULT, help="Max context length (tokens)")
    ap.add_argument("--overlap", type=int, default=OVERLAP_DEFAULT, help="Token overlap between sliding windows")
    ap.add_argument("--small_model", type=str, default=DEFAULT_SMALL, help="Smaller LM (e.g., distilgpt2)")
    ap.add_argument("--large_model", type=str, default=DEFAULT_LARGE, help="Larger LM (e.g., gpt2-medium or Falcon3-3B-Base)")
    args = ap.parse_args()

    # Where to run
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] Device={device}  SMALL={args.small_model}  LARGE={args.large_model}  MAX_CTX={args.max_ctx}")

    # Discover + sample files
    human_paths, ai_paths = load_ghostbuster_paths(args.root, args.domain, args.data_root)
    h_sel, a_sel = balanced_sample(
        human_paths, ai_paths, args.n_per_class, enforce_exact=(not args.no_enforce_exact)
    )
    rows = (
        [{"path": p, "y": 0} for p in h_sel] +
        [{"path": p, "y": 1} for p in a_sel]
    )
    random.shuffle(rows)
    print(f"[info] Using {len(rows)} rows  (ai={sum(r['y']==1 for r in rows)}, human={sum(r['y']==0 for r in rows)})")

    # Read texts + metadata
    texts, labels, paths, hashes = [], [], [], []
    for r in tqdm(rows, desc="Reading texts", unit="file"):
        t = read_text(r["path"])
        texts.append(t)
        labels.append(int(r["y"]))
        paths.append(r["path"])
        hashes.append(sha1(t))

    # --- Two passes: compute NLLs for small, then large ---
    nll_small = compute_nlls_for_model(
        args.small_model, texts, device, args.max_ctx, args.overlap,
        desc=f"NLL pass (small: {args.small_model})"
    )
    nll_large = compute_nlls_for_model(
        args.large_model, texts, device, args.max_ctx, args.overlap,
        desc=f"NLL pass (large: {args.large_model})"
    )

    assert len(nll_small) == len(texts) == len(nll_large)

    # Build unified CSV rows
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dataset = "ghostbuster"
    domain_map = {"wp": "gb-wp", "reuter": "gb-reuter", "essay": "gb-essay"}
    domain_tag = domain_map[args.domain]
    detector = f"binoculars-open:{args.small_model}->{args.large_model}"

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","dataset","domain","detector","y_true","y_score","pred","threshold","text_len","meta"])

        pbar = tqdm(range(len(texts)), desc="Writing CSV", unit="row")
        for i in pbar:
            raw = nll_small[i] - nll_large[i]       # small - large
            y_score = sigmoid(raw)                  # (0,1), higher ⇒ AI
            meta = {
                "path": paths[i],
                "text_hash": hashes[i],
                "nll_small": round(float(nll_small[i]), 6),
                "nll_large": round(float(nll_large[i]), 6),
                "raw_diff": round(float(raw), 6),
                "small_model": args.small_model,
                "large_model": args.large_model,
                "max_ctx": args.max_ctx,
                "overlap": args.overlap,
            }
            row = [
                f"{dataset}_{args.domain}_{i:08d}",
                dataset,
                domain_tag,
                detector,
                int(labels[i]),
                f"{y_score:.6f}",
                "", "",                             # pred, threshold blank
                len(texts[i]),
                json.dumps(meta, ensure_ascii=False)
            ]
            w.writerow(row)

    print(f"[DONE] wrote {args.out}")
    print("Tip: evaluate with your unified evaluator (remember: higher y_score ⇒ AI).")

if __name__ == "__main__":
    main()
