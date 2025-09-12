# binoculars_hc3_std.py
# Run: python binoculars_hc3_std.py --out hc3_binoculars.csv

import os, sys, csv, time, math, argparse
from datetime import datetime, timedelta

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np

# -------------------- Config --------------------
DEFAULT_SMALL = "distilgpt2"
DEFAULT_LARGE = "gpt2-medium"

MAX_CTX = 256
BATCH = 1
CHECKPOINT_EVERY = 100

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
    raise RuntimeError(f"Could not load HC3 {subset_name} ({last_err})")

def iter_rows(limit=None):
    ds = load_hc3("all")
    cnt = 0
    for ex in ds:
        for h in (ex.get("human_answers") or []):
            if h and h.strip():
                yield {"text": h, "label": 0}
                cnt += 1
                if limit and cnt >= limit: return
        for a in (ex.get("chatgpt_answers") or []):
            if a and a.strip():
                yield {"text": a, "label": 1}
                cnt += 1
                if limit and cnt >= limit: return

# ----------------- Core scoring -----------------
def _ensure_min_tokens(input_ids, attn, tok):
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else (tok.pad_token_id or 0)
    if input_ids.size(1) == 0:
        device = input_ids.device
        input_ids = torch.tensor([[eos_id, eos_id]], device=device)
        attn = torch.tensor([[1, 1]], device=device)
    elif input_ids.size(1) == 1:
        device = input_ids.device
        input_ids = torch.cat([input_ids, torch.tensor([[eos_id]], device=device)], dim=1)
        attn = torch.cat([attn, torch.ones((1,1), device=device)], dim=1)
    return input_ids, attn

def avg_nll(model, tok, texts, device, use_amp):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=MAX_CTX)
    input_ids = enc["input_ids"].to(device)
    attn      = enc["attention_mask"].to(device)
    input_ids, attn = _ensure_min_tokens(input_ids, attn, tok)

    with torch.inference_mode():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_ids=input_ids, attention_mask=attn).logits
        else:
            logits = model(input_ids=input_ids, attention_mask=attn).logits

        logits = logits[:, :-1, :]
        labels = input_ids[:, 1:]
        mask   = attn[:, 1:]

        loss_tok = F.cross_entropy(
            logits.transpose(1, 2), labels, reduction="none"
        )
        loss_tok = loss_tok * mask
        tok_counts = mask.sum(dim=1).clamp(min=1)
        nll = (loss_tok.sum(dim=1) / tok_counts)
        return nll.detach().cpu().tolist()

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default=DEFAULT_SMALL)
    ap.add_argument("--large", default=DEFAULT_LARGE)
    ap.add_argument("--out", required=True, help="Output CSV")
    ap.add_argument("--limit", type=int, default=None, help="Limit rows (per class combined)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    print(f"[info] Device={device} small={args.small} large={args.large}")

    tok_s = AutoTokenizer.from_pretrained(args.small, use_fast=True)
    tok_l = AutoTokenizer.from_pretrained(args.large, use_fast=True)
    mdl_s = AutoModelForCausalLM.from_pretrained(args.small).to(device).eval()
    mdl_l = AutoModelForCausalLM.from_pretrained(args.large).to(device).eval()

    rows = list(iter_rows(args.limit))

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","dataset","domain","detector","y_true","y_score","pred","threshold","text_len","meta"])

        pbar = tqdm(rows, desc="Binoculars", unit="row")
        for i, r in enumerate(pbar):
            txt = r["text"].strip()
            y   = r["label"]

            try:
                nll_s = avg_nll(mdl_s, tok_s, [txt], device, use_amp)[0]
                nll_l = avg_nll(mdl_l, tok_l, [txt], device, use_amp)[0]
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("[ERROR] OOM — reduce MAX_CTX or try smaller models.")
                    break
                continue

            # Difference → sigmoid normalization
            raw_score = nll_s - nll_l
            y_score = float(1 / (1 + math.exp(-raw_score)))  # squash into (0,1)

            row = [
                f"hc3_{i:08d}",
                "hc3",
                "hc3",                  # no finer domain
                "binoculars-open",
                y,
                f"{y_score:.6f}",
                "", "",                 # pred, threshold blank
                len(txt),
                "{}"                    # meta blank
            ]
            w.writerow(row)

    print(f"[DONE] wrote {out_path}")

if __name__ == "__main__":
    main()
