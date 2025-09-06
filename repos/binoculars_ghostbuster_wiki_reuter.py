#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, csv, glob, hashlib, random, argparse, math, time, warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.logging import set_verbosity_error
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score

# Quiet HF warnings
set_verbosity_error()
warnings.filterwarnings(
    "ignore",
    message="Token indices sequence length is longer than the specified maximum",
)

# Sensible speed defaults
DEFAULT_SMALL = "distilgpt2"
DEFAULT_LARGE = "gpt2-medium"
DEFAULT_MAX_CTX = 128
DEFAULT_OVERLAP = 8
DEFAULT_BATCH = 8
EPS = 1e-9

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
    return any(glob.glob(p, recursive=True) for p in pats)

def resolve_domain_base(root_dir: str, domain: str, data_root: str | None) -> str:
    candidates = []
    if data_root:
        if os.path.basename(os.path.normpath(data_root)).lower() == domain.lower():
            candidates.append(data_root)
        else:
            candidates.append(os.path.join(data_root, domain))
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
        f"Could not find *.txt files for domain '{domain}'. Tried:\n  - " + "\n  - ".join(tried)
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
    print("#human txt:", len(human_paths), " | #ai txt:", len(ai_paths))
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

def _fmt_eta(secs: float) -> str:
    if secs is None or math.isnan(secs) or math.isinf(secs) or secs < 0:
        return "--:--:--"
    secs = int(secs + 0.5)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# ---------- Tokenization & windowing ----------
def ids_from_text(tok, text: str):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = tok.encode(text or "", add_special_tokens=False)
    if len(ids) < 2:
        ids = ids + [tok.eos_token_id if tok.eos_token_id is not None else (ids[-1] if ids else 0)]
    return ids

def windows_from_ids(ids, max_ctx: int, overlap: int):
    step = max(1, max_ctx - overlap)
    out = []
    for start in range(0, len(ids), step):
        chunk = ids[start:start+max_ctx]
        if len(chunk) < 2: break
        out.append(chunk)
        if start + max_ctx >= len(ids): break
    return out

@torch.inference_mode()
def nll_windows_batched(model, device, windows, batch_size: int):
    total_loss, total_tokens = 0.0, 0
    if not windows: return 0.0, 0
    i = 0
    while i < len(windows):
        batch = windows[i:i+batch_size]
        max_len = max(len(x) for x in batch)
        input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long, device=device)
        attn      = torch.zeros_like(input_ids, dtype=torch.long, device=device)
        for r, seq in enumerate(batch):
            L = len(seq)
            input_ids[r, :L] = torch.tensor(seq, dtype=torch.long, device=device)
            attn[r, :L] = 1
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        logits = logits[:, :-1, :]
        labels = input_ids[:, 1:]
        mask   = attn[:, 1:]
        loss_tok = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
        loss_tok = loss_tok * mask
        total_loss  += float(loss_tok.sum().item())
        total_tokens += int(mask.sum().item())
        i += batch_size
    return total_loss, total_tokens

def nll_one_text(model, tok, device, ids, max_ctx: int, overlap: int, batch_windows: int) -> float:
    windows = windows_from_ids(ids, max_ctx, overlap)
    total_loss, total_tokens = nll_windows_batched(model, device, windows, batch_windows)
    return (total_loss / total_tokens) if total_tokens > 0 else 0.0

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=os.path.abspath("."), help="Project root")
    ap.add_argument("--data_root", type=str, default=None, help="Folder containing essay/reuter/wp or the domain folder directly")
    ap.add_argument("--domain", type=str, required=True, choices=["essay","reuter","wp"])
    ap.add_argument("--n_per_class", type=int, default=1000)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--max_ctx", type=int, default=DEFAULT_MAX_CTX)
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    ap.add_argument("--batch_windows", type=int, default=DEFAULT_BATCH)

    ap.add_argument("--small_model", type=str, default=DEFAULT_SMALL)
    ap.add_argument("--large_model", type=str, default=DEFAULT_LARGE)
    ap.add_argument("--large_on_cpu", action="store_true")
    ap.add_argument("--mode", type=str, default="two_pass", choices=["two_pass","interleaved"])

    # Auto-tune & I/O
    ap.add_argument("--auto_metric", type=str, default="f1", choices=["f1","accuracy","youden"],
                    help="Metric to optimize when choosing threshold.")
    ap.add_argument("--flip_if_better", action="store_true",
                    help="If ROC-AUC improves by flipping the score sign, flip it.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite output instead of resuming (recommended).")
    ap.add_argument("--write_shifted", action="store_true",
                    help="Write y_score = shifted score so that 0.0 == tuned threshold.")
    # Manual threshold (skip auto-tune if set)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Manual threshold on raw score (nll_small - nll_large).")
    # Speed/quantization
    ap.add_argument("--load_8bit", action="store_true", help="Load models in 8-bit (bitsandbytes)")
    ap.add_argument("--load_4bit", action="store_true", help="Load models in 4-bit (bitsandbytes)")
    ap.add_argument("--no_sdpa", action="store_true", help="Disable SDPA attention")
    args = ap.parse_args()

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Devices
    cuda = torch.cuda.is_available()
    small_device = "cuda" if cuda else "cpu"
    large_device = "cpu" if args.large_on_cpu else ("cuda" if cuda else "cpu")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"Small device: {small_device} | Large device: {large_device}")
    print(f"Models: small='{args.small_model}'  large='{args.large_model}'")
    print(f"ctx={args.max_ctx}, overlap={args.overlap}, batch_windows={args.batch_windows}")
    print(f"Quantization: 8bit={args.load_8bit}  4bit={args.load_4bit}  SDPA={'off' if args.no_sdpa else 'on'}")

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.small_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def load_model(name, device):
        kw = {}
        if not args.no_sdpa:
            kw["attn_implementation"] = "sdpa"
        if device == "cuda":
            if args.load_4bit or args.load_8bit:
                try:
                    from transformers import BitsAndBytesConfig
                except Exception:
                    print("[warn] bitsandbytes not installed; ignoring --load_4bit/--load_8bit")
                else:
                    if args.load_4bit:
                        bnb = BitsAndBytesConfig(
                            load_in_4bit=True, bnb_4bit_use_double_quant=True,
                            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
                        )
                        return AutoModelForCausalLM.from_pretrained(
                            name, device_map="auto", quantization_config=bnb, low_cpu_mem_usage=True, **kw
                        ).eval()
                    if args.load_8bit:
                        bnb = BitsAndBytesConfig(load_in_8bit=True)
                        return AutoModelForCausalLM.from_pretrained(
                            name, device_map="auto", quantization_config=bnb, low_cpu_mem_usage=True, **kw
                        ).eval()
            mdl = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch.float16, low_cpu_mem_usage=True, **kw
            ).to(device).eval()
            try: torch.set_float32_matmul_precision("high")
            except Exception: pass
            return mdl
        else:
            return AutoModelForCausalLM.from_pretrained(name, low_cpu_mem_usage=True, **kw).to(device).eval()

    # Data (always build fresh item list; we overwrite unless told otherwise)
    human_paths, ai_paths = load_ghostbuster_paths(args.root, args.domain, args.data_root)
    h_sel, a_sel = balanced_sample(human_paths, ai_paths, args.n_per_class)
    rows = (
        [{"path": p, "label": 0, "src": f"GB-{args.domain}", "kind": "human"} for p in h_sel] +
        [{"path": p, "label": 1, "src": f"GB-{args.domain}", "kind": "ai"} for p in a_sel]
    )
    random.shuffle(rows)

    items = []
    for r in tqdm(rows, desc="Reading & tokenizing", unit="file"):
        text = read_text(r["path"])
        thash = sha1(text)
        ids = ids_from_text(tok, text)
        items.append((r, thash, ids))

    # Scoring passes
    @torch.inference_mode()
    def compute_all_nlls(name, device, tag):
        nll_map = {}
        mdl = load_model(name, device)
        pbar = tqdm(items, desc=f"{tag} pass ({name} on {device})", unit="file")
        ema = None; alpha = 0.3
        for (r, thash, ids) in pbar:
            t0 = time.perf_counter()
            try:
                nll = nll_one_text(mdl, tok, device, ids, args.max_ctx, args.overlap, args.batch_windows)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    print("\n[ERROR] CUDA OOM. Try --max_ctx 96/64, --batch_windows 4/2, or quantize.")
                    sys.exit(1)
                nll = float("nan")
            except Exception:
                nll = float("nan")
            nll_map[thash] = nll
            dt = time.perf_counter() - t0
            ema = dt if ema is None else (alpha*dt + (1-alpha)*ema)
            remain = len(items) - len(nll_map)
            pbar.set_postfix({"r/s": f"{(1.0/(ema or 1e-9)):.2f}", "ETA": _fmt_eta(remain*(ema or 0.0))})
        del mdl
        if device == "cuda": torch.cuda.empty_cache()
        return nll_map

    nll_small = compute_all_nlls(args.small_model, "cuda" if torch.cuda.is_available() else "cpu", "small")
    nll_large = compute_all_nlls(args.large_model, "cuda" if torch.cuda.is_available() and not args.large_on_cpu else "cpu", "large")

    # Build arrays
    y_true, s_raw, nll_s_arr, nll_l_arr, keep = [], [], [], [], []
    for (r, thash, ids) in items:
        a = nll_small.get(thash, float("nan"))
        b = nll_large.get(thash, float("nan"))
        if math.isnan(a) or math.isnan(b):
            continue
        y_true.append(int(r["label"]))
        s_raw.append(a - b)
        nll_s_arr.append(a)
        nll_l_arr.append(b)
        keep.append((r, thash))

    y_true = np.array(y_true, dtype=int)
    s = np.array(s_raw, dtype=float)

    # Optional sign flip
    if args.flip_if_better and len(np.unique(y_true)) == 2:
        try:
            auc_pos = roc_auc_score(y_true, s)
            auc_neg = roc_auc_score(y_true, -s)
            if auc_neg > auc_pos:
                s = -s
                print(f"[auto] Flipped score sign (AUC {auc_neg:.3f} > {auc_pos:.3f})")
        except Exception:
            pass

    # Threshold: manual or auto sweep
    if args.threshold is not None:
        thr = float(args.threshold)
        metric_name = "manual"
        yhat = (s >= thr).astype(int)
        best_f1  = f1_score(y_true, yhat, zero_division=0)
        best_acc = accuracy_score(y_true, yhat)
    else:
        uniq = np.unique(s[~np.isnan(s)])
        if len(uniq) > 4000:
            idx = np.linspace(0, len(uniq)-1, 4000, dtype=int)
            thr_grid = uniq[idx]
        else:
            thr_grid = uniq
        best = {"thr": None, "val": -1.0, "acc": None, "f1": None}
        for t in thr_grid:
            yhat = (s >= t).astype(int)
            acc  = accuracy_score(y_true, yhat)
            f1   = f1_score(y_true, yhat, zero_division=0)
            if args.auto_metric == "accuracy":
                score_val = acc
            elif args.auto_metric == "youden":
                tn = ((y_true==0)&(yhat==0)).sum()
                fp = ((y_true==0)&(yhat==1)).sum()
                fn = ((y_true==1)&(yhat==0)).sum()
                tp = ((y_true==1)&(yhat==1)).sum()
                tpr = tp/(tp+fn) if (tp+fn)>0 else 0
                tnr = tn/(tn+fp) if (tn+fp)>0 else 0
                score_val = tpr + tnr - 1
            else:
                score_val = f1
            if score_val > best["val"]:
                best = {"thr": float(t), "val": float(score_val), "acc": float(acc), "f1": float(f1)}
        thr = best["thr"]; best_f1 = best["f1"]; best_acc = best["acc"]; metric_name = args.auto_metric

    # Threshold-free metrics
    try:
        roc_auc = roc_auc_score(y_true, s)
        pr_auc  = average_precision_score(y_true, s)
    except Exception:
        roc_auc = float("nan"); pr_auc = float("nan")

    pos_rate = float((s >= thr).mean())
    print("\n=== Binoculars summary ===")
    print(f" ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")
    print(f" Best-{metric_name} threshold={thr:.6f}  F1={best_f1:.4f}  ACC={best_acc:.4f}  pos-rate={pos_rate:.3f}")

    # Write output (OVERWRITE by default)
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[WARN] {out_path} exists. Use --overwrite to replace. Writing *_new.csv instead.")
        base, ext = os.path.splitext(out_path)
        out_path = f"{base}_new{ext}"

    header = ["path","text_hash","label","nll_small","nll_large","score_raw","y_score","pred","src","kind"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for (i, ((r, thash))) in enumerate(keep):
            a = nll_s_arr[i]; b = nll_l_arr[i]
            score_raw = s[i]
            y_score = score_raw - thr if args.write_shifted else score_raw
            pred = int((y_score if args.write_shifted else score_raw) >= (0.0 if args.write_shifted else thr))
            w.writerow([
                r["path"], thash, r["label"],
                f"{a:.6f}", f"{b:.6f}",
                f"{score_raw:.6f}", f"{y_score:.6f}",
                pred, r["src"], r["kind"]
            ])

    print(f"[DONE] Wrote {len(keep)} rows → {out_path}")
    print("Columns: score_raw = (nll_small - nll_large) [maybe sign-flipped], y_score = score_raw - threshold (if --write_shifted).")
    print("Your evaluator that thresholds at 0.0 should read y_score when --write_shifted is used.")
    if not args.write_shifted:
        print("Tip: pass --write_shifted so 0.0 equals the tuned threshold and you avoid 'all 1s'.")
    
if __name__ == "__main__":
    main()
