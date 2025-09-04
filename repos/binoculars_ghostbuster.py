# binoculars_ghostbuster.py
import os, sys, csv, glob, hashlib
import random
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------- Defaults tuned for ~4 GB VRAM --------------------
SMALL = "distilgpt2"
LARGE = "gpt2-medium"
MAX_CTX = 256          # try 256; drop to 192/128 if OOM
BATCH = 1              # 1 is safest for VRAM
CHECKPOINT_EVERY = 100

# -------------------- Robust file reading --------------------
def read_text(path):
    # Be robust to odd encodings:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")

def sha1(s):
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

# -------------------- Dataset loader (tailored to your tree) --------------------
def load_ghostbuster_paths(root_dir, domain):
    """
    Expected layout (as provided in your tree):
      <root>/data/ghostbuster-data/<domain>/
        human/**.txt
        gpt/**.txt
        gpt_prompt1/**.txt
        gpt_prompt2/**.txt
        gpt_semantic/**.txt
        gpt_writing/**.txt
        claude/**.txt
    """
    base = os.path.join(root_dir, "data", "ghostbuster-data", domain)

    human_dir = "human"
    ai_dirs_map = {
        "essay":  ["gpt", "gpt_prompt1", "gpt_prompt2", "gpt_semantic", "gpt_writing", "claude"],
        "reuter": ["gpt", "gpt_prompt1", "gpt_prompt2", "gpt_semantic", "gpt_writing", "claude"],
        "wp":     ["gpt", "gpt_prompt1", "gpt_prompt2", "gpt_semantic", "gpt_writing", "claude"],
    }
    if domain not in ai_dirs_map:
        raise ValueError(f"Unknown domain: {domain}")

    # collect human
    human_glob = os.path.join(base, human_dir, "**", "*.txt")
    human_paths = sorted(glob.glob(human_glob, recursive=True))

    # collect AI union
    ai_paths = []
    for d in ai_dirs_map[domain]:
        ai_paths.extend(glob.glob(os.path.join(base, d, "**", "*.txt"), recursive=True))
    ai_paths = sorted(ai_paths)

    print("Looking for data under:", base)
    print("#human txt:", len(human_paths), "#ai txt:", len(ai_paths))

    if not human_paths or not ai_paths:
        subdirs = []
        if os.path.isdir(base):
            try:
                subdirs = next(os.walk(base))[1]
            except StopIteration:
                pass
        raise FileNotFoundError(
            f"No files found for one or both classes under {base}.\n"
            f"human={len(human_paths)}, ai={len(ai_paths)}\n"
            f"Expected AI dirs: {', '.join(ai_dirs_map[domain])}\n"
            f"Found subdirs: {', '.join(subdirs)}"
        )
    return human_paths, ai_paths

def balanced_sample(human_paths, ai_paths, n_per_class, seed=42):
    random.seed(seed)
    random.shuffle(human_paths)
    random.shuffle(ai_paths)
    h_sel = human_paths[:n_per_class]
    a_sel = ai_paths[:n_per_class]
    return h_sel, a_sel

# -------------------- NLL scorer --------------------
def avg_nll(model, tok, texts, device, use_amp, max_ctx):
    # Ensure pad token exists
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- CRITICAL FIX: sanitize empty inputs so T>0 ----
    safe_texts = []
    eos = tok.eos_token or " "
    for t in texts:
        s = (t or "").strip()
        if not s:
            s = eos            # ensure at least one token
        safe_texts.append(s)

    enc = tok(
        safe_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_ctx,
    )
    input_ids = enc["input_ids"].to(device, non_blocking=True)
    attn      = enc["attention_mask"].to(device, non_blocking=True)

    with torch.inference_mode():
        if use_amp:
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(input_ids=input_ids, attention_mask=attn).logits
        else:
            logits = model(input_ids=input_ids, attention_mask=attn).logits

        # shift
        logits = logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()
        mask   = attn[:, 1:].contiguous()

        # If a sequence was exactly 1 token after sanitize, T-1==0; guard that too.
        if logits.shape[1] == 0:
            # return 0 loss for such degenerate cases (rare after sanitize)
            return [0.0 for _ in range(input_ids.shape[0])]

        loss_tok = F.cross_entropy(
            logits.transpose(1, 2),  # [B,V,T-1]
            labels, reduction="none"
        )
        loss_tok = loss_tok * mask
        tok_counts = mask.sum(dim=1).clamp(min=1)
        nll = (loss_tok.sum(dim=1) / tok_counts)  # per-example avg NLL
        return nll.detach().cpu().tolist()

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=os.path.abspath("."), help="Project root that contains data/ghostbuster-data")
    ap.add_argument("--domain", type=str, required=True, choices=["essay","reuter","wp"], help="Ghostbuster domain")
    ap.add_argument("--n_per_class", type=int, default=1000, help="Balanced samples per class")
    ap.add_argument("--out", type=str, required=True, help="Output CSV path")
    ap.add_argument("--max_ctx", type=int, default=MAX_CTX, help="Max context length (tokens)")
    ap.add_argument("--batch", type=int, default=BATCH, help="Batch size (1 recommended for ~4 GB VRAM)")
    args = ap.parse_args()

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    print(f"Device: {device}")

    # Tokenizers & models
    tok_s = AutoTokenizer.from_pretrained(SMALL, use_fast=True)
    tok_l = AutoTokenizer.from_pretrained(LARGE, use_fast=True)
    if tok_s.pad_token is None: tok_s.pad_token = tok_s.eos_token
    if tok_l.pad_token is None: tok_l.pad_token = tok_l.eos_token

    if device == "cuda":
        # transformers>=4.44 warns about torch_dtype; 'dtype' is preferred but both still work
        mdl_s = AutoModelForCausalLM.from_pretrained(SMALL, dtype=torch.float16, device_map="auto").eval()
        mdl_l = AutoModelForCausalLM.from_pretrained(LARGE, dtype=torch.float16, device_map="auto").eval()
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        mdl_s = AutoModelForCausalLM.from_pretrained(SMALL).eval()
        mdl_l = AutoModelForCausalLM.from_pretrained(LARGE).eval()

    # Load file paths and pick a balanced subset
    human_paths, ai_paths = load_ghostbuster_paths(args.root, args.domain)
    n_per = min(args.n_per_class, len(human_paths), len(ai_paths))
    if n_per < args.n_per_class:
        print(f"[WARN] Reducing n_per_class to {n_per} due to available files.")
    h_sel, a_sel = balanced_sample(human_paths, ai_paths, n_per)

    rows = [{"path": p, "label": 0, "src": f"GB-{args.domain}", "kind": "human"} for p in h_sel] + \
           [{"path": p, "label": 1, "src": f"GB-{args.domain}", "kind": "ai"}    for p in a_sel]
    random.shuffle(rows)

    # Resume support
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

    # Main scoring loop
    wrote_since_flush = 0
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        pbar = tqdm(rows, desc=f"Scoring GB-{args.domain} (ctx={args.max_ctx}, bs={args.batch})")
        batch_texts, batch_meta = [], []

        def flush_batch():
            nonlocal batch_texts, batch_meta, wrote_since_flush
            if not batch_texts:
                return
            texts = batch_texts
            try:
                nll_s = avg_nll(mdl_s, tok_s, texts, next(mdl_s.parameters()).device, use_amp, args.max_ctx)
                nll_l = avg_nll(mdl_l, tok_l, texts, next(mdl_l.parameters()).device, use_amp, args.max_ctx)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("[ERROR] CUDA OOM. Try reducing --max_ctx to 192 or 128 and/or --batch to 1.")
                    sys.exit(1)
                raise
            for meta, a, b, t in zip(batch_meta, nll_s, nll_l, texts):
                score = a - b
                pred  = int(score > 0)  # naive; calibrate later
                w.writerow([meta["path"], meta["text_hash"], meta["label"],
                            f"{a:.6f}", f"{b:.6f}", f"{score:.6f}", pred, meta["src"], meta["kind"]])
                wrote_since_flush += 1
            batch_texts, batch_meta = [], []

        for r in pbar:
            if r["path"] in done_keys:
                continue
            text = read_text(r["path"])
            thash = sha1(text)
            if thash in done_keys:
                continue

            batch_texts.append(text)
            batch_meta.append({"path": r["path"], "text_hash": thash, "label": r["label"],
                               "src": r["src"], "kind": r["kind"]})

            if len(batch_texts) >= args.batch:
                flush_batch()
                if wrote_since_flush >= CHECKPOINT_EVERY:
                    f.flush()
                    os.fsync(f.fileno())
                    wrote_since_flush = 0
        # flush tail
        flush_batch()
        f.flush()
        os.fsync(f.fileno())
    print(f"[DONE] Wrote {out_path}")

if __name__ == "__main__":
    main()
