# binoculars_hc3_twopass.py
# Run:
#   python binoculars_hc3_twopass.py --out hc3_binoculars.csv
#   (optional) --small distilgpt2 --large tiiuae/falcon-7b-instruct --limit 6000
#
# Two-pass Binoculars:
#   1) Load SMALL, compute NLL_small for all texts, free SMALL.
#   2) Load LARGE, compute NLL_large for all texts, free LARGE.
#   3) y_score = sigmoid(NLL_small - NLL_large)
#
# Unified CSV: id,dataset,domain,detector,y_true,y_score,pred,threshold,text_len,meta
# - dataset="hc3"
# - domain="hc3-all"
# - detector=f"binoculars-<small>-<large>"
# - y_score in [0,1]; pred/threshold left blank.

import os, sys, csv, math, argparse, json
from datetime import datetime, timedelta
from typing import List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import numpy as np
import random

# -------------------- Defaults --------------------
DEFAULT_SMALL = "distilgpt2"
DEFAULT_LARGE = "gpt2-medium"

MAX_CTX = 256
BATCH = 1  # keep 1 for low VRAM
CHECKPOINT_EVERY = 200

# ----------------- HC3 loader -----------------
def load_hc3(subset_name: str = "all"):
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
    raise RuntimeError(f"Could not load HC3 subset '{subset_name}' ({last_err})")

def collect_hc3(limit_total: int | None, seed: int = 42) -> Tuple[List[str], List[int]]:
    """
    Return texts and labels (0 human, 1 ai).
    If limit_total is provided (e.g., 6000), we target half per class (e.g., 3000/3000).
    Falls back to min available per class if not enough.
    """
    ds = load_hc3("all")
    humans, ais = [], []
    for ex in ds:
        for h in (ex.get("human_answers") or []):
            if h and isinstance(h, str) and h.strip():
                humans.append(h)
        for a in (ex.get("chatgpt_answers") or []):
            if a and isinstance(a, str) and a.strip():
                ais.append(a)

    random.seed(seed)
    # Target balanced sampling if limit_total given
    if limit_total and limit_total > 0:
        per_class = limit_total // 2
        k = min(per_class, len(humans), len(ais))
        random.shuffle(humans); random.shuffle(ais)
        humans = humans[:k]
        ais    = ais[:k]
    # else: use all available

    texts = humans + ais
    labels = [0]*len(humans) + [1]*len(ais)

    # Shuffle to mix classes
    idx = list(range(len(texts)))
    random.shuffle(idx)
    texts  = [texts[i] for i in idx]
    labels = [labels[i] for i in idx]
    return texts, labels

# ----------------- Core scoring -----------------
def _ensure_min_tokens(input_ids, attn, tok):
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else (tok.pad_token_id or 0)
    device = input_ids.device
    if input_ids.size(1) == 0:
        input_ids = torch.tensor([[eos_id, eos_id]], device=device, dtype=torch.long)
        attn      = torch.tensor([[1, 1]], device=device, dtype=torch.long)
    elif input_ids.size(1) == 1:
        input_ids = torch.cat([input_ids, torch.tensor([[eos_id]], device=device, dtype=torch.long)], dim=1)
        attn      = torch.cat([attn, torch.ones((1,1), device=device, dtype=torch.long)], dim=1)
    return input_ids, attn

@torch.inference_mode()
def avg_nll(model, tok, text: str, max_ctx: int) -> float:
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_ctx)
    input_ids = enc["input_ids"].to(model.device, dtype=torch.long, non_blocking=True)
    attn      = enc["attention_mask"].to(model.device, dtype=torch.long, non_blocking=True)
    input_ids, attn = _ensure_min_tokens(input_ids, attn, tok)

    use_amp = (model.device.type == "cuda")
    if use_amp:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(input_ids=input_ids, attention_mask=attn).logits
    else:
        logits = model(input_ids=input_ids, attention_mask=attn).logits

    logits = logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous()
    mask   = attn[:, 1:].contiguous()

    loss_tok = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
    loss_tok = loss_tok * mask
    tok_counts = mask.sum(dim=1).clamp(min=1)
    nll = (loss_tok.sum(dim=1) / tok_counts).item()
    return float(nll)

def sigmoid(x: float) -> float:
    # numerically stable-ish
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def compute_nlls_for_model(model_id: str, texts: List[str], device: str) -> Tuple[list, dict]:
    """
    Loads a single model, computes avg NLL for each text, and returns list of NLLs + meta.
    """
    print(f"[load] {model_id} on {device}")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if device == "cuda":
        mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, trust_remote_code=True).to("cuda").eval()
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        mdl = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval()

    nlls = []
    pbar = tqdm(total=len(texts), desc=f"NLL({model_id})", unit="doc")
    for i, txt in enumerate(texts):
        t = (txt or "").strip()
        try:
            nll = avg_nll(mdl, tok, t, MAX_CTX)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if device == "cuda":
                    torch.cuda.empty_cache()
                raise
            # Skip pathological rows
            nll = float("nan")
        nlls.append(nll)
        if (i+1) % CHECKPOINT_EVERY == 0:
            pbar.set_postfix({"last_nll": f"{nll:.3f}"})
        pbar.update(1)
    pbar.close()

    # Clean up this model fully before loading the next
    del mdl
    if device == "cuda":
        torch.cuda.empty_cache()

    meta = {"model_id": model_id, "max_ctx": MAX_CTX}
    return nlls, meta

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default=DEFAULT_SMALL, help="HF model id for SMALL (e.g., distilgpt2)")
    ap.add_argument("--large", default=DEFAULT_LARGE, help="HF model id for LARGE (e.g., gpt2-medium)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--limit", type=int, default=6000, help="Total rows target (balanced if possible). Default: 6000")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling/shuffle")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] Device={device}  SMALL={args.small}  LARGE={args.large}  MAX_CTX={MAX_CTX}")

    # 1) Collect data (balanced if possible)
    texts, labels = collect_hc3(limit_total=(args.limit if args.limit and args.limit > 0 else None), seed=args.seed)
    n = len(texts)
    n_pos = sum(labels)
    print(f"[info] Using {n} rows  (ai={n_pos}, human={n-n_pos})")

    # 2) Pass 1: SMALL NLLs
    nll_small, meta_small = compute_nlls_for_model(args.small, texts, device)

    # 3) Pass 2: LARGE NLLs
    nll_large, meta_large = compute_nlls_for_model(args.large, texts, device)

    # 4) Compose y_score = sigmoid(nll_small - nll_large)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    det_name = f"binoculars-{args.small.replace('/','_')}-{args.large.replace('/','_')}"
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","dataset","domain","detector","y_true","y_score","pred","threshold","text_len","meta"])

        for i, (txt, y, ns, nl) in enumerate(zip(texts, labels, nll_small, nll_large)):
            # Skip rows where any NLL is NaN
            if not (isinstance(ns, float) and isinstance(nl, float)) or (math.isnan(ns) or math.isnan(nl)):
                continue
            raw = ns - nl
            y_score = sigmoid(raw)
            meta = {
                "nll_small": ns,
                "nll_large": nl,
                "small": meta_small["model_id"],
                "large": meta_large["model_id"],
                "max_ctx": MAX_CTX,
            }
            w.writerow([
                f"hc3_{i:08d}",
                "hc3",
                "hc3-all",
                det_name,
                int(y),
                f"{y_score:.6f}",
                "", "",                    # pred, threshold blank
                len((txt or "")),
                json.dumps(meta, ensure_ascii=False),
            ])

    print(f"[DONE] wrote {args.out}")

if __name__ == "__main__":
    main()
