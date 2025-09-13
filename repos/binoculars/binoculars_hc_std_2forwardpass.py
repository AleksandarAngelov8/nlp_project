# binoculars_two_pass_hc3.py
# Usage (Colab):
#   !pip install -q "transformers>=4.41.0" accelerate bitsandbytes datasets tqdm einops
#   !python binoculars_two_pass_hc3.py --out /content/hc3_binoculars_falcon7b_2pass.csv
#
# Notes:
# - Pass 1 loads ONLY the small model, computes nll_small, writes temp CSV.
# - Pass 2 unloads small, loads ONLY the large model (Falcon-7B in 4-bit), computes nll_large, then writes final CSV.
# - Score = sigmoid(nll_small - nll_large). Higher ⇒ more AI-like.
# - Outputs unified schema: id,dataset,domain,detector,y_true,y_score,pred,threshold,text_len,meta

import os, csv, math, json, argparse, gc
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------- defaults (edit if you like) -------------
DEFAULT_SMALL = "distilgpt2"
DEFAULT_LARGE = "tiiuae/falcon-7b-instruct"  # or "tiiuae/falcon-7b"
MAX_CTX = 256        # lower to 192/128 if OOM
SEED = 42
ENFORCE_3K = True    # enforce 3000 human + 3000 ai if available

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def load_hc3_all():
    last = None
    for ext in ("jsonl","json"):
        try:
            fp = hf_hub_download("Hello-SimpleAI/HC3", filename=f"all.{ext}", repo_type="dataset")
            return load_dataset("json", data_files=fp, split="train")
        except Exception as e:
            last = e
    raise RuntimeError(f"HC3 load failed: {last}")

def collect_hc3_rows() -> Tuple[List[str], List[str]]:
    ds = load_hc3_all()
    humans, ais = [], []
    for ex in ds:
        for h in (ex.get("human_answers") or []):
            if isinstance(h, str) and h.strip():
                humans.append(h)
        for a in (ex.get("chatgpt_answers") or []):
            if isinstance(a, str) and a.strip():
                ais.append(a)
    return humans, ais

def ensure_min_tokens(input_ids, attn, tok):
    eos = tok.eos_token_id if tok.eos_token_id is not None else (tok.pad_token_id or 0)
    dev = input_ids.device
    if input_ids.size(1) == 0:
        input_ids = torch.tensor([[eos, eos]], device=dev, dtype=torch.long)
        attn      = torch.tensor([[1, 1]], device=dev, dtype=torch.long)
    elif input_ids.size(1) == 1:
        input_ids = torch.cat([input_ids, torch.tensor([[eos]], device=dev, dtype=torch.long)], dim=1)
        attn      = torch.cat([attn, torch.ones((1,1), device=dev, dtype=torch.long)], dim=1)
    return input_ids, attn

@torch.inference_mode()
def avg_nll(model, tok, text: str, max_ctx: int) -> float:
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_ctx)
    input_ids = enc["input_ids"].to(model.device, dtype=torch.long, non_blocking=True)
    attn      = enc["attention_mask"].to(model.device, dtype=torch.long, non_blocking=True)
    input_ids, attn = ensure_min_tokens(input_ids, attn, tok)

    use_amp = (model.device.type == "cuda")
    if use_amp:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(input_ids=input_ids, attention_mask=attn).logits
    else:
        logits = model(input_ids=input_ids, attention_mask=attn).logits

    logits = logits[:, :-1, :]
    labels = input_ids[:, 1:]
    mask   = attn[:, 1:]
    loss_tok = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
    loss_tok = loss_tok * mask
    tok_count = mask.sum(dim=1).clamp(min=1)
    nll = (loss_tok.sum(dim=1) / tok_count).item()
    return float(nll)

def pass1_small_nll(temp_csv: Path, small_id: str, max_ctx: int):
    # Load SMALL on GPU (fp16)
    tok_s = AutoTokenizer.from_pretrained(small_id, use_fast=True)
    small = AutoModelForCausalLM.from_pretrained(small_id, torch_dtype=torch.float16).to("cuda").eval()

    import numpy as np
    humans, ais = collect_hc3_rows()
    print(f"[info] HC3 pool: human={len(humans)} ai={len(ais)}")

    rng = np.random.default_rng(SEED)
    if ENFORCE_3K and len(humans) >= 3000 and len(ais) >= 3000:
        humans = list(rng.choice(humans, size=3000, replace=False))
        ais    = list(rng.choice(ais,    size=3000, replace=False))
        print("[info] presampled to 3k/3k")
    else:
        print("[warn] insufficient for strict 3k/3k; using all available")

    texts = humans + ais
    labels = [0]*len(humans) + [1]*len(ais)
    idx = rng.permutation(len(texts))
    texts = [texts[i] for i in idx]
    labels = [labels[i] for i in idx]

    temp_csv.parent.mkdir(parents=True, exist_ok=True)
    with temp_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","text","y_true","text_len","nll_small"])
        pbar = tqdm(total=len(texts), desc="Pass1 small NLL", unit="doc")
        for i, (txt, y) in enumerate(zip(texts, labels)):
            txt = (txt or "").strip()
            nll_s = avg_nll(small, tok_s, txt, max_ctx)
            w.writerow([f"hc3_{i:08d}", txt, int(y), len(txt), f"{nll_s:.6f}"])
            pbar.update(1)
    print(f"[pass1] wrote {temp_csv}")

    # Free small
    del small; del tok_s
    gc.collect(); torch.cuda.empty_cache()

def pass2_large_and_finalize(temp_csv: Path, out_csv: Path, large_id: str, max_ctx: int):
    # Load LARGE in 4-bit with offloading (Colab-safe)
    from transformers import BitsAndBytesConfig
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    max_memory = {"cuda:0": "12GiB", "cpu": "24GiB"}
    tok_l = AutoTokenizer.from_pretrained(large_id, use_fast=True, trust_remote_code=True)
    if tok_l.pad_token is None:
        tok_l.pad_token = tok_l.eos_token
    large = AutoModelForCausalLM.from_pretrained(
        large_id,
        trust_remote_code=True,
        quantization_config=bnb_cfg,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory=max_memory,
        offload_folder=str(out_csv.parent / "offload")
    ).eval()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with temp_csv.open("r", encoding="utf-8") as fi, out_csv.open("w", newline="", encoding="utf-8") as fo:
        r = csv.DictReader(fi)
        w = csv.writer(fo)
        det = f"binoculars-{DEFAULT_SMALL.replace('/','_')}-{large_id.replace('/','_')}"
        w.writerow(["id","dataset","domain","detector","y_true","y_score","pred","threshold","text_len","meta"])

        pbar = tqdm(total=sum(1 for _ in open(temp_csv, "r", encoding="utf-8")) - 1, desc="Pass2 large NLL", unit="doc")
        fi.seek(0); next(fi)  # skip header

        for line in fi:
            row = next(csv.reader([line]))
            # row fields: id,text,y_true,text_len,nll_small
            rid, txt, y_true, text_len, nll_s = row[0], row[1], int(row[2]), int(row[3]), float(row[4])
            nll_l = avg_nll(large, tok_l, txt, max_ctx)
            raw = nll_s - nll_l
            y_score = sigmoid(raw)
            meta = {"nll_small": nll_s, "nll_large": nll_l, "small": DEFAULT_SMALL, "large": large_id, "max_ctx": max_ctx}
            w.writerow([rid, "hc3", "hc3-all", det, y_true, f"{y_score:.6f}", "", "", text_len, json.dumps(meta, ensure_ascii=False)])
            pbar.update(1)

    print(f"[pass2] wrote {out_csv}")

    # Free large
    del large; del tok_l
    gc.collect(); torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default=DEFAULT_SMALL, help="small LM (e.g., distilgpt2)")
    ap.add_argument("--large", default=DEFAULT_LARGE, help="large LM (e.g., tiiuae/falcon-7b-instruct)")
    ap.add_argument("--out", required=True, help="final CSV path")
    ap.add_argument("--tmp", default="/content/_hc3_binoculars_pass1.csv", help="temp CSV between passes")
    ap.add_argument("--max_ctx", type=int, default=MAX_CTX)
    args = ap.parse_args()

    tmp = Path(args.tmp)
    out = Path(args.out)

    pass1_small_nll(tmp, args.small, args.max_ctx)
    pass2_large_and_finalize(tmp, out, args.large, args.max_ctx)

if __name__ == "__main__":
    main()
