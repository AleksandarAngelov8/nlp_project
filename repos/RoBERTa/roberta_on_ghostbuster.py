# -*- coding: utf-8 -*-
import os, sys, csv, glob, hashlib, random, argparse, time, math
from datetime import datetime, timedelta
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "Hello-SimpleAI/chatgpt-detector-roberta"

# ---------- robust file reading ----------
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

def make_key(path: str, label: int, kind: str, text_hash: str) -> str:
    raw = f"{label}|{kind}|{text_hash}|{os.path.basename(path)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

# ---------- base dir resolution ----------
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
        # if they gave the domain dir directly, use as-is; else join domain
        if os.path.basename(os.path.normpath(data_root)).lower() == domain.lower():
            candidates.append(data_root)
        else:
            candidates.append(os.path.join(data_root, domain))
    # common layouts
    candidates += [
        os.path.join(root_dir, "data", "ghostbuster-data", domain),
        os.path.join(root_dir, "ghostbuster", "data", "ghostbuster-data", domain),
        os.path.join(root_dir, "ghostbuster-data", domain),
        os.path.join(root_dir, domain),
    ]
    tried = []
    for c in candidates:
        tried.append(c)
        if os.path.isdir(c) and _has_txt(c):
            return c
    # If none had .txt but one exists as a dir, check for logprobs-only (likely missing LFS)
    existing_dirs = [c for c in candidates if os.path.isdir(c)]
    hint = ""
    if existing_dirs:
        # look for lots of logprobs dirs
        has_only_logprobs = False
        for d in existing_dirs:
            lp = glob.glob(os.path.join(d, "**", "logprobs"), recursive=True)
            if lp and not _has_txt(d):
                has_only_logprobs = True
                break
        if has_only_logprobs:
            hint = (
                "\nIt looks like you may have only the 'logprobs/' folders but not the *.txt files. "
                "If you cloned with Git, make sure Git LFS is installed and pulled:\n"
                "    git lfs install\n"
                "    git lfs pull\n"
                "Or download the dataset with LFS-enabled zip from GitHub."
            )
    raise FileNotFoundError(
        "Could not find any *.txt files for domain '{d}'. Tried:\n  - "
        + "\n  - ".join(tried)
        + hint.replace("{d}", domain)
    )

# ---------- dataset discovery ----------
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

# ---------- inference ----------
def batched_predict(texts, tok, model, device, max_len, use_amp=False):
    enc = tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
    enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
    with torch.inference_mode():
        if use_amp and device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**enc).logits
        else:
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
    return probs.detach().cpu().numpy()

def _fmt_eta(secs: float) -> str:
    if secs is None or math.isnan(secs) or math.isinf(secs) or secs < 0:
        return "--:--:--"
    secs = int(secs + 0.5)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=os.path.abspath("."), help="Project root (default: current dir)")
    ap.add_argument("--data_root", type=str, default=None,
                    help="Path to 'ghostbuster-data' root (folder that contains essay/reuter/wp) "
                         "or directly to the domain folder (…/ghostbuster-data/essay).")
    ap.add_argument("--domain", type=str, required=True, choices=["essay","reuter","wp"], help="Ghostbuster domain")
    ap.add_argument("--n_per_class", type=int, default=1000, help="Balanced samples per class")
    ap.add_argument("--out", type=str, required=True, help="Output CSV path")
    ap.add_argument("--max_len", type=int, default=512, help="Tokenizer max_length")
    ap.add_argument("--batch", type=int, default=16, help="Batch size (reduce if CPU-only)")
    ap.add_argument("--min_chars", type=int, default=0, help="Skip texts shorter than this many characters")
    args = ap.parse_args()

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    print("Device:", device)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, use_safetensors=True
    ).to(device).eval()
    torch.set_grad_enabled(False)

    # discover & sample
    human_paths, ai_paths = load_ghostbuster_paths(args.root, args.domain, args.data_root)
    h_sel, a_sel = balanced_sample(human_paths, ai_paths, args.n_per_class)
    rows = (
        [{"path": p, "label": 0, "src": f"GB-{args.domain}", "kind": "human"} for p in h_sel] +
        [{"path": p, "label": 1, "src": f"GB-{args.domain}", "kind": "ai"} for p in a_sel]
    )
    random.shuffle(rows)

    # resume support (content-aware)
    done_keys = set()
    if os.path.exists(out_path):
        print(f"[INFO] Resuming from {out_path}")
        df_done = pd.read_csv(out_path)
        if "key" in df_done.columns:
            done_keys = set(df_done["key"].astype(str).tolist())
    else:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["key","path","text_hash","label","p_human","p_ai","pred","src","kind"])

    # progress/ETA
    CHECKPOINT_EVERY = 200
    wrote_since_flush = 0
    start = time.perf_counter()
    ema_s_per_row = None
    ema_alpha = 0.2
    processed_new = 0

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        batch_texts, batch_meta = [], []
        pbar = tqdm(rows, desc=f"RoBERTa on GB-{args.domain} (batch={args.batch})", unit="file")

        for r in pbar:
            txt = read_text(r["path"])
            if args.min_chars > 0 and len(txt) < args.min_chars:
                continue
            th = sha1(txt)
            k = make_key(r["path"], r["label"], r["kind"], th)
            if k in done_keys:
                continue

            batch_texts.append(txt)
            batch_meta.append((r, th, k))

            if len(batch_texts) >= args.batch:
                t0 = time.perf_counter()
                probs = batched_predict(batch_texts, tok, model, device, args.max_len, use_amp=use_amp)
                dt = time.perf_counter() - t0

                for (meta, thash, key), pr in zip(batch_meta, probs):
                    p_h, p_a = float(pr[0]), float(pr[1])
                    pred = int(p_a >= 0.5)
                    w.writerow([key, meta["path"], thash, meta["label"],
                                f"{p_h:.6f}", f"{p_a:.6f}", pred, meta["src"], meta["kind"]])
                    wrote_since_flush += 1
                    processed_new += 1
                    done_keys.add(key)

                # ETA
                rows_in_batch = len(batch_texts)
                s_per_row = dt / max(rows_in_batch, 1)
                if ema_s_per_row is None:
                    ema_s_per_row = s_per_row
                else:
                    ema_s_per_row = ema_alpha * s_per_row + (1 - ema_alpha) * ema_s_per_row
                # rough remaining = total rows - processed_new
                remained = max(len(rows) - processed_new, 0)
                eta_sec = remained * ema_s_per_row
                pbar.set_postfix({
                    "r/s": f"{(1.0/ema_s_per_row):.2f}" if ema_s_per_row else "0.00",
                    "avg_s/row": f"{ema_s_per_row:.3f}" if ema_s_per_row else "0.000",
                    "ETA": _fmt_eta(eta_sec),
                    "fin": (datetime.now() + timedelta(seconds=eta_sec)).strftime("%H:%M:%S"),
                })

                batch_texts, batch_meta = [], []

                if wrote_since_flush >= CHECKPOINT_EVERY:
                    f.flush()
                    os.fsync(f.fileno())
                    wrote_since_flush = 0

        # flush tail
        if batch_texts:
            probs = batched_predict(batch_texts, tok, model, device, args.max_len, use_amp=use_amp)
            for (meta, thash, key), pr in zip(batch_meta, probs):
                p_h, p_a = float(pr[0]), float(pr[1])
                pred = int(p_a >= 0.5)
                w.writerow([key, meta["path"], thash, meta["label"],
                            f"{p_h:.6f}", f"{p_a:.6f}", pred, meta["src"], meta["kind"]])
            f.flush()
            os.fsync(f.fileno())

    elapsed = time.perf_counter() - start
    print(f"[DONE] Wrote {out_path} | processed={processed_new} | elapsed {elapsed:.1f}s")

if __name__ == "__main__":
    main()
