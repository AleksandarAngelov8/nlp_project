#!/usr/bin/env python3
# eval_raid_open.py — Ghostbuster (open-LM) scoring on RAID CSV
# - Input: RAID CSV (e.g., train.csv / extra.csv)
# - Text column: --text_col (default: generation)
# - Labels: use --label_col if present; else derive from 'model' (non-empty => AI=1)
# - Outputs per row: id, domain, attack, model, decoding, repetition_penalty, source_id, adv_source_id,
#                    y_true, y_score, detector_name, text_len, [pred_label, threshold_used if threshold>=0]
# - Requires your trained Ghostbuster artifacts: model/, features.txt, trigram_model.pkl

import os, csv, time, argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import dill as pickle
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from utils.featurize import t_featurize_logprobs, score_ngram
from utils.symbolic import train_trigram, get_words, vec_functions, scalar_functions

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate Ghostbuster (open-LM) on RAID CSV.")
    ap.add_argument("--csv", required=True, help="Path to RAID CSV (train/extra or any RAID-like file).")
    ap.add_argument("--text_col", default="generation",
                    help="Column containing the text to score (default: generation).")
    ap.add_argument("--label_col", default=None,
                    help="Column with ground-truth labels (0/1 or human/ai). If omitted, derive from 'model'.")
    ap.add_argument("--assume_label", type=int, default=None,
                    help="Force label for ALL rows (0=human,1=ai). Overrides label_col/model derivation.")
    ap.add_argument("--id_col", default="id", help="Row identifier column (default: id).")
    ap.add_argument("--limit", type=int, default=-1, help="Max rows to process; -1 = all.")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle rows before limiting.")
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"], help="Force device; default: auto.")
    ap.add_argument("--model", default=os.getenv("GB_OPEN_LM", "gpt2-medium"),
                    help="HF causal LM (e.g., gpt2, gpt2-medium). Smaller = faster on CPU.")
    ap.add_argument("--max_ctx", type=int, default=int(os.getenv("GB_OPEN_MAX_CTX", "768")),
                    help="Max tokens for open LM (reduce on CPU, e.g., 384/256).")
    ap.add_argument("--threshold", type=float, default=-1.0,
                    help="If >=0, write pred_label and threshold_used using this decision threshold.")
    ap.add_argument("--out", default="outputs/ghostbuster_open/raid_scores.csv",
                    help="Output CSV path.")
    return ap.parse_args()

# ---------- Utilities ----------
def fmt_eta(seconds: float) -> str:
    if not seconds or seconds < 0 or np.isinf(seconds) or np.isnan(seconds):
        return "--:--:--"
    seconds = int(seconds + 0.5)
    return f"{seconds//3600:02d}:{(seconds%3600)//60:02d}:{seconds%60:02d}"

def normalize_labels(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.strip().str.lower()
    mapping = {"human":"0","ai":"1","bot":"1","machine":"1","model":"1",
               "0":"0","1":"1","true":"1","false":"0"}
    s = s.map(lambda x: mapping.get(x, x))
    out = pd.to_numeric(s, errors="coerce").fillna(0).astype(int).values
    return np.clip(out, 0, 1)

def derive_labels_from_model(series: pd.Series) -> np.ndarray:
    labels = []
    for v in series:
        if pd.isna(v):
            labels.append(0)
        else:
            s = str(v).strip().lower()
            labels.append(0 if s in {"", "nan", "none", "human"} else 1)
    return np.array(labels, dtype=int)

# ---------- Ghostbuster feature pipeline ----------
@torch.inference_mode()
def token_probs_and_subwords(text: str, tok, mdl, max_ctx: int):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_ctx)
    input_ids = enc["input_ids"].to(mdl.device)
    attn      = enc["attention_mask"].to(mdl.device)
    # ensure at least one next-token to score
    if input_ids.shape[1] < 2:
        eos_id = tok.eos_token_id if tok.eos_token_id is not None else (tok.pad_token_id or 0)
        input_ids = torch.cat([input_ids, torch.tensor([[eos_id]], device=mdl.device)], dim=1)
        attn      = torch.cat([attn, torch.ones_like(attn)], dim=1)
    out = mdl(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logits = out.logits
    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    next_ids = input_ids[:, 1:].unsqueeze(-1)
    tok_logp = torch.gather(logprobs, dim=-1, index=next_ids).squeeze(0).squeeze(-1)
    probs = torch.exp(tok_logp).detach().cpu().numpy()
    subwords = tok.convert_ids_to_tokens(input_ids[0, 1:].tolist())
    return probs, subwords

def build_feature_vector(text: str, best_features: List[str],
                         trigram_model, enc_for_trigram,
                         tok, mdl, max_ctx: int) -> np.ndarray:
    probs, subwords = token_probs_and_subwords(text, tok, mdl, max_ctx)
    trigram_arr = np.array(score_ngram(text, trigram_model,      enc_for_trigram, n=3, strip_first=False))
    unigram_arr = np.array(score_ngram(text, trigram_model.base, enc_for_trigram, n=1, strip_first=False))
    t_features = t_featurize_logprobs(probs, probs, subwords)  # alias both GPT streams to open-LM probs

    vector_map = {
        "davinci-logprobs": probs,
        "ada-logprobs":     probs,
        "trigram-logprobs": trigram_arr,
        "unigram-logprobs": unigram_arr,
    }
    exp_features = []
    for exp in best_features:
        tokens = get_words(exp)
        curr = vector_map[tokens[0]]
        i = 1
        while i < len(tokens):
            tk = tokens[i]
            if tk in vec_functions:
                nxt = vector_map[tokens[i+1]]
                curr = vec_functions[tk](curr, nxt)
                i += 2
            elif tk in scalar_functions:
                exp_features.append(scalar_functions[tk](curr))
                break
            else:
                break
    return np.array(t_features + exp_features, dtype=float)

# ---------- Scoring loop ----------
def score_rows(df: pd.DataFrame,
               text_col: str,
               id_col: str,
               y_true: np.ndarray,
               best_features: List[str],
               trigram_model,
               enc_for_trigram,
               tok, mdl, max_ctx: int,
               out_path: Path,
               threshold: Optional[float]) -> Tuple[int, float]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    use_threshold = (threshold is not None) and (threshold >= 0.0)

    # RAID metadata columns we’ll emit if present:
    meta_cols = ["domain", "attack", "model", "decoding", "repetition_penalty",
                 "source_id", "adv_source_id"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["id"] + meta_cols + ["y_true", "y_score", "detector_name", "text_len"]
        if use_threshold:
            header += ["pred_label", "threshold_used"]
        w.writerow(header)
        print(f"[info] Writing: {out_path}")

        # Progress/ETA
        n = len(df)
        ema_row_sec = None
        alpha = 0.2
        start_t = time.perf_counter()
        pbar = tqdm(total=n, desc="RAID scoring", unit="doc")

        wrote = 0
        for i, row in enumerate(df.itertuples(index=False)):
            txt = str(getattr(row, text_col, "") or "").strip()
            if not txt:
                # still write a row with y_score=0.0 to keep alignment
                yscore = 0.0
            else:
                feats = build_feature_vector(txt, best_features, trigram_model, enc_for_trigram, tok, mdl, max_ctx)
                yscore = MODEL.predict_proba(((feats - MU) / SIGMA).reshape(1, -1))[:, 1][0]

            # gather IDs/metas
            rid = str(getattr(row, id_col, f"raid_{i:08d}"))
            meta_vals = [getattr(row, c, "") if hasattr(row, c) else "" for c in meta_cols]
            y = int(y_true[i])
            out_row = [rid] + meta_vals + [y, f"{yscore:.6f}", "ghostbuster-open", len(txt)]
            if use_threshold:
                yhat = int(yscore >= threshold)
                out_row += [yhat, f"{threshold:.6f}"]
            w.writerow(out_row)

            # ETA updates
            t1 = time.perf_counter()
            dt = t1 - start_t if wrote == 0 else t1 - last_t
            last_t = t1
            ema_row_sec = dt if ema_row_sec is None else (alpha*dt + (1-alpha)*ema_row_sec)
            left = n - (i + 1)
            eta = left * (ema_row_sec or 0.0)
            pbar.set_postfix({
                "r/s": f"{(1.0/(ema_row_sec or 1e-9)):.2f}",
                "avg_s/row": f"{(ema_row_sec or 0):.3f}",
                "ETA": fmt_eta(eta),
                "fin": (datetime.now() + timedelta(seconds=eta)).strftime("%H:%M:%S")
            })
            pbar.update(1)
            wrote += 1

        pbar.close()
        total_dt = time.perf_counter() - start_t
        rps = wrote / total_dt if total_dt > 0 else 0.0
        print(f"[done] wrote {wrote} rows in {fmt_eta(total_dt)} (~{rps:.2f} r/s)")
        return wrote, rps

# ---------- Main ----------
def main():
    global MODEL, MU, SIGMA  # used in score_rows()
    args = parse_args()

    # Load GB artifacts
    MODEL = pickle.load(open("model/model", "rb"))
    MU    = pickle.load(open("model/mu", "rb"))
    SIGMA = pickle.load(open("model/sigma", "rb"))
    best_features = Path("model/features.txt").read_text(encoding="utf-8").strip().split("\n")

    # Device & open-LM
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Device: {device} | OPEN_LM={args.model} | MAX_CTX={args.max_ctx}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    try:
        if device == "cuda":
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Trigram model & encoder
    import tiktoken
    trigram_model = pickle.load(open("model/trigram_model.pkl", "rb"))
    enc_for_trigram = tiktoken.encoding_for_model("davinci").encode

    # Load RAID CSV
    print(f"[info] Loading CSV: {args.csv}")
    try:
        df = pd.read_csv(args.csv, engine="pyarrow")
    except Exception:
        df = pd.read_csv(args.csv)

    if args.shuffle:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    if args.limit and args.limit > 0 and args.limit < len(df):
        df = df.iloc[:args.limit].reset_index(drop=True)

    # Validate columns and build labels
    text_col = args.text_col
    if text_col not in df.columns:
        # try common fallbacks
        for c in ["response", "text", "output", "content"]:
            if c in df.columns:
                text_col = c
                print(f"[warn] text_col '{args.text_col}' not found; using '{text_col}'")
                break
        else:
            raise ValueError(f"Could not find text column. Tried '{args.text_col}' and common fallbacks. Columns: {list(df.columns)}")

    # Determine labels
    if args.assume_label is not None:
        y_true = np.full(len(df), int(args.assume_label), dtype=int)
        print(f"[info] Forcing all labels to {args.assume_label} due to --assume_label.")
    elif args.label_col and args.label_col in df.columns:
        y_true = normalize_labels(df[args.label_col])
        print(f"[info] Using explicit label column: {args.label_col}")
    elif "label" in df.columns:
        y_true = normalize_labels(df["label"])
        print("[info] Using 'label' column.")
    elif "is_human" in df.columns:
        y_true = normalize_labels(df["is_human"])  # mapping will flip to 0/1
        print("[info] Using 'is_human' column.")
    elif "model" in df.columns:
        y_true = derive_labels_from_model(df["model"])
        print("[info] No explicit labels; derived from 'model' (non-empty => AI=1).")
    else:
        raise ValueError("No labels found/derivable. Provide --label_col or --assume_label, "
                         "or ensure a 'model' column is present to derive labels.")

    # ID column
    id_col = args.id_col if args.id_col in df.columns else None
    if id_col is None:
        print(f"[warn] id_col '{args.id_col}' not found; will auto-generate IDs.")

    # Score & write
    wrote, rps = score_rows(
        df=df,
        text_col=text_col,
        id_col=(id_col or ""),
        y_true=y_true,
        best_features=best_features,
        trigram_model=trigram_model,
        enc_for_trigram=enc_for_trigram,
        tok=tok, mdl=mdl, max_ctx=args.max_ctx,
        out_path=Path(args.out),
        threshold=(args.threshold if args.threshold is not None and args.threshold >= 0.0 else None),
    )

    print(f"[DONE] rows={wrote}  out={args.out}")

if __name__ == "__main__":
    main()
