# roberta_raid_std.py
# Unified CSV schema:
# id,dataset,domain,detector,y_true,y_score,pred,threshold,text_len,meta

import os, math, argparse, time, sys
from typing import Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub.utils import RepositoryNotFoundError

from std_eval import StdCollector  # unified CSV + (optional) balancing

MODEL_NAME = "Hello-SimpleAI/chatgpt-detector-roberta"

# --- Hardening: avoid buggy header paths & excessive hub chatter ---
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
for bad in ("HF_HUB_HEADERS", "HUGGINGFACE_HUB_HEADERS"):
    if os.environ.get(bad):
        print(f"[WARN] Clearing malformed env var: {bad}")
        os.environ.pop(bad, None)

# ---------- helpers ----------
def _fmt_eta(secs: Optional[float]) -> str:
    if secs is None or math.isnan(secs) or math.isinf(secs) or secs < 0:
        return "--:--:--"
    secs = int(secs + 0.5)
    h, rem = divmod(secs, 3600); m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def normalize_split(name: str) -> str:
    n = (name or "").strip().lower()
    if n in {"val", "val_labeled", "valid", "validation"}:
        print(f"[INFO] Mapping split '{name}' -> 'extra'")
        return "extra"
    return n

def pick_text_col(df: pd.DataFrame) -> str:
    for c in ["generation","response","text","output","content"]:
        if c in df.columns: return c
    raise ValueError(f"No text column found. Looked for generation/response/text/output/content. Got: {list(df.columns)}")

def pick_label_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["label","is_human","target","y","gt"]:
        if c in df.columns: return c
    if "model" in df.columns:
        print("[INFO] Using 'model' to derive labels (non-empty => AI).")
        return "model"
    return None

def normalize_labels_from_model_column(series: pd.Series) -> np.ndarray:
    out=[]
    for v in series:
        if pd.isna(v): out.append(0); continue
        s=str(v).strip().lower()
        out.append(0 if s in {"","nan","none","human"} else 1)
    return np.array(out, dtype=int)

def normalize_labels(series: pd.Series) -> np.ndarray:
    if series.name == "model":  # derive from model column
        return normalize_labels_from_model_column(series)
    s = series.astype(str).str.strip().str.lower()
    mapping = {"human":"0","ai":"1","bot":"1","machine":"1","model":"1","0":"0","1":"1"}
    s = s.map(lambda x: mapping.get(x, x))
    out = pd.to_numeric(s, errors="coerce").fillna(0).astype(int).values
    return np.clip(out, 0, 1)

def batched_predict(texts, tok, model, device, max_len, use_amp=False):
    enc = tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
    enc = {k: v.to(device, non_blocking=True) for k,v in enc.items()}
    with torch.inference_mode():
        if use_amp and device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**enc).logits
        else:
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)  # [p_human, p_ai]
    return probs.detach().cpu().numpy()

# ---------- local loading helpers ----------
def _raid_default_cache(split: str) -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "raid", f"{split}.csv")

def _read_any_table(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, engine="pyarrow")
    except Exception:
        return pd.read_csv(path)

def _maybe_local_raid_df(split: str, data_dir: Optional[str]) -> Optional[pd.DataFrame]:
    cand=[]
    if data_dir:
        for ext in (".parquet",".csv",".csv.gz",".csv.zst"):
            cand.append(os.path.join(data_dir, f"{split}{ext}"))
    cand.append(_raid_default_cache(split))
    for p in cand:
        if os.path.isfile(p):
            print(f"[INFO] Loading local RAID file: {p}")
            return _read_any_table(p)
    return None

# ---------- data loading ----------
def load_raid_labeled(split_in: str, data_dir: Optional[str] = None) -> pd.DataFrame:
    split = normalize_split(split_in)
    df_local = _maybe_local_raid_df(split, data_dir)
    if df_local is not None:
        print("[INFO] Loaded RAID from local disk.")
        return df_local

    # raid-bench API
    try:
        from raid.utils import load_data  # type: ignore
        print(f"[INFO] Trying raid-bench API: load_data(split='{split}')")
        df = load_data(split=split)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        print("[INFO] Loaded via raid-bench API.")
        return df
    except Exception as e:
        print(f"[WARN] raid-bench API failed ({e}). Falling back to HF…")

    # Hugging Face datasets mirror
    try:
        from datasets import load_dataset  # type: ignore
        print(f"[INFO] Loading HF dataset: liamdugan/raid split='{split}'")
        ds = load_dataset("liamdugan/raid", split=split)
        df = ds.to_pandas()
        print("[INFO] Loaded via Hugging Face datasets.")
        return df
    except Exception as e2:
        raise RuntimeError(f"Could not load RAID split '{split_in}'. Provide --csv/--data_dir. HF error: {e2}")

# ---------- model loading (robust) ----------
def load_detector_model(model_name: str, device: str):
    """
    Robust loader that avoids the safetensors auto-conversion path that can
    trigger header issues in some environments.
    """
    if os.path.isdir(model_name):
        raise OSError(
            f"A local directory named '{model_name}' exists and shadows the HF repo.\n"
            f"Rename or remove that folder, or cd elsewhere."
        )
    # Prefer non-safetensors to avoid auto-conversion probe
    try:
        mdl = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            use_safetensors=False,
            local_files_only=False,
            trust_remote_code=False,
        ).to(device).eval()
        return mdl
    except RepositoryNotFoundError as e:
        raise FileNotFoundError(
            f"HF repo not found: {model_name}. Are you online / authenticated?"
        ) from e
    except Exception as e1:
        print("[WARN] Non-safetensors load failed, retrying with safetensors. Error:", repr(e1))
        mdl = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            use_safetensors=True,
            local_files_only=False,
            trust_remote_code=False,
        ).to(device).eval()
        return mdl

# ---------- presampling helpers ----------
def stratified_presample_indices(y: np.ndarray, k_per_class: int, rng_seed: int = 42) -> Optional[np.ndarray]:
    """Return indices for exactly k_per_class per class if available; else None."""
    idx_h = np.where(y == 0)[0]
    idx_a = np.where(y == 1)[0]
    if len(idx_h) < k_per_class or len(idx_a) < k_per_class:
        return None
    rng = np.random.default_rng(rng_seed)
    sel_h = rng.choice(idx_h, size=k_per_class, replace=False)
    sel_a = rng.choice(idx_a, size=k_per_class, replace=False)
    sel = np.concatenate([sel_h, sel_a])
    rng.shuffle(sel)
    return sel

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="Local labeled CSV/Parquet (overrides --split).")
    ap.add_argument("--data_dir", type=str, default=None, help="Folder with RAID files (train/extra).")
    ap.add_argument("--split", type=str, default="train", help="RAID split with labels: train or extra.")
    ap.add_argument("--sample", type=int, default=0, help="Random sample size; 0 or negative = all rows (ignored if enforcing 3k).")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--out_csv", type=str, required=True, help="Unified CSV output path.")
    ap.add_argument("--enforce_3k", action="store_true", default=True, help="Require 3000 per class (default ON).")
    ap.add_argument("--strict_fail", action="store_true", default=False, help="If <3k per class and --enforce_3k, hard fail.")
    ap.add_argument("--log_pred_at", type=float, default=None, help="Optional threshold to also populate pred/threshold.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    print("Device:", device)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    try:
        model = load_detector_model(MODEL_NAME, device)
    except Exception as e:
        print("[ERROR] Could not load detector model:", repr(e), file=sys.stderr)
        raise

    torch.set_grad_enabled(False)

    # Load data
    if args.csv:
        if not os.path.isfile(args.csv):
            raise FileNotFoundError(f"--csv not found: {args.csv}")
        print(f"[INFO] Using explicit local file: {args.csv}")
        df = _read_any_table(args.csv)
        domain = "raid-custom"
        split_tag = "custom"
    else:
        df = load_raid_labeled(args.split, data_dir=args.data_dir)
        split_norm = normalize_split(args.split)
        domain = f"raid-{split_norm}"
        split_tag = split_norm

    print("Columns:", list(df.columns))
    text_col = pick_text_col(df)
    label_col = pick_label_col(df)
    if label_col is None:
        raise ValueError("No labels available (and no 'model' column). Provide a labeled CSV or a labeled split.")

    y = normalize_labels(df[label_col])
    texts = df[text_col].astype(str).tolist()

    n_h = int((y == 0).sum()); n_a = int((y == 1).sum())
    print(f"[INFO] Label counts: human={n_h}, ai={n_a}")

    # --- presample to exactly 3k/3k BEFORE inference whenever possible ---
    presel = None
    if args.enforce_3k:
        presel = stratified_presample_indices(y, 3000, rng_seed=42)
        if presel is None:
            msg = f"[WARN] Not enough examples for 3k/3k (human={n_h}, ai={n_a})."
            if args.strict_fail:
                raise RuntimeError(msg + " Use --strict_fail False or collect more data.")
            print(msg, "Proceeding with ALL available data (CSV will have <6000 rows).")

    # If we presampled, use only those indices; else optional random sample
    if presel is not None:
        base_idx = presel.tolist()
        print(f"[INFO] Presampling to exactly 6000 rows (3k/3k) before inference.")
    else:
        base_idx = list(range(len(texts)))
        if args.sample and args.sample > 0 and args.sample < len(base_idx):
            np.random.seed(42)
            base_idx = np.random.choice(len(base_idx), size=args.sample, replace=False).tolist()
            print(f"[INFO] Random pre-sample to {len(base_idx)} rows (no class balancing).")

    print(f"Evaluating {len(base_idx)} rows from {domain}")

    # Collector: if we've presampled to 3k/3k already, we can skip internal balancing
    collector = StdCollector(
        dataset="raid",
        domain=domain,
        detector="roberta-hc3",
        split=split_tag,
        random_state=42,
        enforce_3k_per_class=False if presel is not None else bool(args.enforce_3k)
    )

    start = time.perf_counter()
    ema_s_per_row = None; ema_alpha = 0.2

    batch_texts, batch_indices = [], []
    pbar = tqdm(range(len(base_idx)), desc=f"RoBERTa on {domain} (batch={args.batch})", unit="ex")

    def flush_batch():
        nonlocal batch_texts, batch_indices, ema_s_per_row
        if not batch_texts: return
        t0 = time.perf_counter()
        probs = batched_predict(batch_texts, tok, model, device, args.max_len, use_amp=use_amp)
        dt = time.perf_counter() - t0

        for i_row, p in zip(batch_indices, probs):
            src_idx = base_idx[i_row]
            p_a = float(p[1])   # y_score = P(ai)
            txt = texts[src_idx]
            collector.add(
                idx=src_idx,
                y_true=int(y[src_idx]),
                y_score=p_a,
                text_len=len(txt),
                meta={"source_row": int(src_idx), "text_col": text_col, "model": MODEL_NAME}
            )

        rows_in_batch = len(batch_texts)
        s_per_row = dt / max(rows_in_batch, 1)
        ema_s_per_row = s_per_row if ema_s_per_row is None else (ema_alpha*s_per_row + (1-ema_alpha)*ema_s_per_row)
        remained = max(len(base_idx) - (batch_indices[-1] + 1), 0)
        eta_sec = remained * (ema_s_per_row or 0)
        pbar.set_postfix({
            "r/s": f"{(1.0/(ema_s_per_row or 1e-9)):.2f}",
            "avg_s/row": f"{(ema_s_per_row or 0):.3f}",
            "ETA": _fmt_eta(eta_sec),
            "fin": (datetime.now() + timedelta(seconds=eta_sec)).strftime("%H:%M:%S") if eta_sec>0 else "--:--:--",
        })
        batch_texts, batch_indices = [], []

    for i in pbar:
        batch_texts.append(texts[base_idx[i]])
        batch_indices.append(i)
        if len(batch_texts) >= args.batch:
            flush_batch()
    flush_batch()

    # Finalize unified CSV; optionally also write pred/threshold
    thr = args.log_pred_at if args.log_pred_at is not None else None
    info = collector.finalize(out_csv=args.out_csv, known_flip=False, threshold=thr)
    elapsed = time.perf_counter() - start

    print(f"[DONE] Unified CSV: {info['out']}  n_total={info['n_total']}  n_written={info['n_written']}  flipped={info['flipped']}")
    print(f"Elapsed: {elapsed:.1f}s")

if __name__ == "__main__":
    main()
