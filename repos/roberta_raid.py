# roberta_raid.py
# Runs Hello-SimpleAI/chatgpt-detector-roberta on a labeled RAID split.
# Prefers local files first (--csv or --data_dir, or ~/.cache/raid/{split}.csv),
# otherwise tries raid-bench API, then Hugging Face Datasets.
#
# Output CSV columns (compatible with eval_csv.py):
#   id, label, p_human, p_ai, pred, text
#
# Usage examples:
#   python roberta_raid.py --split train --sample 5000 --out_csv outputs/raid/roberta_train5k.csv
#   python roberta_raid.py --split extra --batch 32 --max_len 512 --out_csv outputs/raid/roberta_extra_all.csv
#   python roberta_raid.py --csv D:\datasets\raid\train.csv --out_csv outputs/raid/roberta_train.csv
#   python roberta_raid.py --split train --data_dir D:\datasets\raid --out_csv outputs/raid/roberta_train.csv
#
# Notes:
# - Valid RAID splits: "train", "extra" (no "val"). "test" is unlabeled by design.
# - If explicit label column is missing, labels are derived from RAID's "model" column
#   (any non-empty model name => AI=1; empty/NaN/"human" => 0).

import os
import math
import argparse
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "Hello-SimpleAI/chatgpt-detector-roberta"

# ---------- helpers ----------
def _fmt_eta(secs: float | None) -> str:
    if secs is None or math.isnan(secs) or math.isinf(secs) or secs < 0:
        return "--:--:--"
    secs = int(secs + 0.5)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def normalize_split(name: str) -> str:
    n = (name or "").strip().lower()
    if n in {"val", "val_labeled", "valid", "validation"}:
        print(f"[INFO] Mapping split '{name}' -> 'extra'")
        return "extra"
    return n  # expected: 'train', 'extra' (note: 'test' is unlabeled)

def pick_text_col(df: pd.DataFrame) -> str:
    for c in ["generation", "response", "text", "output", "content"]:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find a text column in CSV. Looked for: "
        f"['generation','response','text','output','content']. Found: {list(df.columns)}"
    )

def pick_label_col(df: pd.DataFrame) -> str | None:
    for c in ["label", "is_human", "target", "y", "gt"]:
        if c in df.columns:
            return c
    # RAID: if explicit labels absent, use 'model' to derive (non-empty => AI)
    if "model" in df.columns:
        print("[INFO] Using 'model' column to derive labels (non-empty => AI).")
        return "model"
    return None

def normalize_labels_from_model_column(series: pd.Series) -> np.ndarray:
    labels = []
    for val in series:
        if pd.isna(val):
            labels.append(0)  # human
        else:
            val_str = str(val).strip().lower()
            labels.append(0 if val_str in {"", "nan", "none", "human"} else 1)
    return np.array(labels, dtype=int)

def normalize_labels(series: pd.Series) -> np.ndarray:
    # Auto-detect RAID's 'model' column
    if series.name == "model":
        return normalize_labels_from_model_column(series)
    # Otherwise standard mapping
    s = series.astype(str).str.strip().str.lower()
    mapping = {"human": "0", "ai": "1", "bot": "1", "machine": "1", "model": "1", "0": "0", "1": "1"}
    s = s.map(lambda x: mapping.get(x, x))
    out = pd.to_numeric(s, errors="coerce").fillna(0).astype(int).values
    return np.clip(out, 0, 1)

def batched_predict(texts, tok, model, device, max_len, use_amp=False):
    enc = tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
    enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
    with torch.inference_mode():
        if use_amp and device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**enc).logits
        else:
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)  # [B,2] -> [p_human, p_ai]
    return probs.detach().cpu().numpy()

# ---------- local loading helpers ----------
def _raid_default_cache(split: str) -> str:
    # Matches the cache path shown in many raid-bench installs on Windows
    return os.path.join(os.path.expanduser("~"), ".cache", "raid", f"{split}.csv")

def _read_any_table(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    # prefer pyarrow if available
    try:
        return pd.read_csv(path, engine="pyarrow")
    except Exception:
        return pd.read_csv(path)

def _maybe_local_raid_df(split: str, data_dir: str | None) -> pd.DataFrame | None:
    candidates = []
    if data_dir:
        for ext in (".parquet", ".csv", ".csv.gz", ".csv.zst"):
            candidates.append(os.path.join(data_dir, f"{split}{ext}"))
    # Also try existing raid-bench cache
    candidates.append(_raid_default_cache(split))

    for p in candidates:
        if os.path.isfile(p):
            print(f"[INFO] Loading local RAID file: {p}")
            return _read_any_table(p)
    return None

# ---------- data loading ----------
def load_raid_labeled(split_in: str, data_dir: str | None = None) -> pd.DataFrame:
    """
    Prefer local files; otherwise try raid-bench (labeled), then HF mirror.
    Valid splits: 'train', 'extra' (no 'val'; 'test' is unlabeled).
    """
    split = normalize_split(split_in)

    # 0) Local files
    df_local = _maybe_local_raid_df(split, data_dir)
    if df_local is not None:
        print("[INFO] Loaded RAID data from local disk.")
        return df_local

    # 1) raid-bench API
    rb_err = None
    try:
        from raid.utils import load_data  # type: ignore
        print(f"[INFO] Trying raid-bench API: load_data(split='{split}')")
        df = load_data(split=split)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        print("[INFO] Successfully loaded data via raid-bench API.")
        return df
    except Exception as e:
        rb_err = e
        print(f"[WARN] raid-bench API failed ({e}). Falling back to Hugging Face…")

    # 2) Hugging Face datasets
    hf_err = None
    try:
        from datasets import load_dataset  # type: ignore
        print(f"[INFO] Loading HF dataset: liamdugan/raid split='{split}'")
        ds = load_dataset("liamdugan/raid", split=split)  # 'train' or 'extra'
        df = ds.to_pandas()
        print("[INFO] Successfully loaded data via Hugging Face datasets.")
        return df
    except Exception as e2:
        hf_err = e2

    raise RuntimeError(
        f"Could not load RAID labeled split '{split_in}' via local, raid-bench, or HF.\n"
        f"Original errors:\n  raid-bench: {rb_err}\n  HF datasets: {hf_err}\n"
        "Use --split train/extra or provide --csv/--data_dir."
    )

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None,
                    help="Path to a local labeled CSV/Parquet (overrides split).")
    ap.add_argument("--data_dir", type=str, default=None,
                    help="Folder with local RAID files (train/extra as CSV/Parquet).")
    ap.add_argument("--split", type=str, default="train",
                    help="RAID split with labels: 'train' or 'extra' (no 'val').")
    ap.add_argument("--sample", type=int, default=5000,
                    help="Random sample size; 0 or negative = use all rows.")
    ap.add_argument("--batch", type=int, default=16, help="Batch size.")
    ap.add_argument("--max_len", type=int, default=512, help="Tokenizer max_length.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Threshold for AI class (p_ai >= thr => pred=1).")
    ap.add_argument("--assume_label", type=int, default=None,
                    help="If unlabeled, force 0=human or 1=ai to allow downstream eval.")
    ap.add_argument("--out_csv", type=str, required=True, help="Output CSV path.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    print("Device:", device)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, use_safetensors=True
    ).to(device).eval()
    torch.set_grad_enabled(False)

    # Load data (local CSV beats everything; else data_dir/local cache; else online)
    if args.csv:
        if not os.path.isfile(args.csv):
            raise FileNotFoundError(f"--csv not found: {args.csv}")
        print(f"[INFO] Using explicit local file: {args.csv}")
        df = _read_any_table(args.csv)
    else:
        df = load_raid_labeled(args.split, data_dir=args.data_dir)

    print("Columns:", list(df.columns))
    print("DataFrame shape:", df.shape)
    try:
        # Small peek (won't crash if cols missing)
        cols = [c for c in ["model", "generation"] if c in df.columns]
        if cols:
            print("Sample rows:\n" + df[cols].head(3).to_string())
    except Exception:
        pass

    text_col = pick_text_col(df)
    label_col = pick_label_col(df)

    if label_col is None:
        if args.assume_label is None:
            raise ValueError(
                "No label column found and no 'model' column to derive labels.\n"
                "Provide a labeled CSV via --csv, or --assume_label 0|1."
            )
        print(f"[INFO] Using --assume_label={args.assume_label} for all rows.")
        labels = np.full(len(df), int(args.assume_label), dtype=int)
    else:
        labels = normalize_labels(df[label_col])

    # Debugging: label distribution
    uniq, cnt = np.unique(labels, return_counts=True)
    print(f"Label distribution: {dict(zip(uniq, cnt))} (0=human, 1=AI)")

    texts_all = df[text_col].astype(str).tolist()
    ids = df["id"].astype(str).tolist() if "id" in df.columns else [str(i) for i in range(len(df))]
    rows = list(zip(ids, texts_all, labels))

    # sampling
    if args.sample and args.sample > 0 and args.sample < len(rows):
        np.random.seed(42)
        idx = np.random.choice(len(rows), size=args.sample, replace=False)
        rows = [rows[i] for i in idx]

    print(f"Evaluating {len(rows)} rows from split='{normalize_split(args.split)}'")
    start = time.perf_counter()
    ema_s_per_row = None
    ema_alpha = 0.2

    tmp_out = args.out_csv + ".tmp"
    with open(tmp_out, "w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["id", "label", "p_human", "p_ai", "pred", "text"])

        batch_texts, batch_meta = [], []
        pbar = tqdm(rows, desc=f"RoBERTa on RAID[{normalize_split(args.split)}] (batch={args.batch})", unit="ex")
        processed = 0

        def flush_batch():
            nonlocal batch_texts, batch_meta, processed, ema_s_per_row
            if not batch_texts:
                return
            t0 = time.perf_counter()
            probs = batched_predict(batch_texts, tok, model, device, args.max_len, use_amp=use_amp)
            dt = time.perf_counter() - t0

            for (rid, lbl, txt), p in zip(batch_meta, probs):
                p_h, p_a = float(p[0]), float(p[1])
                pred = int(p_a >= args.threshold)
                w.writerow([rid, int(lbl), f"{p_h:.6f}", f"{p_a:.6f}", pred, txt])

            rows_in_batch = len(batch_texts)
            s_per_row = dt / max(rows_in_batch, 1)
            ema_s_per_row = s_per_row if ema_s_per_row is None else (ema_alpha * s_per_row + (1 - ema_alpha) * ema_s_per_row)
            processed += rows_in_batch
            remained = max(len(rows) - processed, 0)
            eta_sec = remained * (ema_s_per_row or 0)
            pbar.set_postfix({
                "r/s": f"{(1.0/(ema_s_per_row or 1e-9)):.2f}",
                "avg_s/row": f"{(ema_s_per_row or 0):.3f}",
                "ETA": _fmt_eta(eta_sec),
                "fin": (datetime.now() + timedelta(seconds=eta_sec)).strftime("%H:%M:%S")
                       if eta_sec > 0 else "--:--:--",
            })
            batch_texts, batch_meta = [], []

        for rid, txt, lbl in pbar:
            batch_texts.append(txt)
            batch_meta.append((rid, lbl, txt))
            if len(batch_texts) >= args.batch:
                flush_batch()
        flush_batch()

    os.replace(tmp_out, args.out_csv)
    elapsed = time.perf_counter() - start
    print(f"[DONE] Wrote {args.out_csv}  |  rows={len(rows)}  |  elapsed={elapsed:.1f}s")
    print("Now you can evaluate:")
    print(f"  python eval_csv.py {args.out_csv} roberta_raid")

if __name__ == "__main__":
    main()
