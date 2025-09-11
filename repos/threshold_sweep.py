#!/usr/bin/env python3
"""
threshold_sweep.py

Sweep decision thresholds over detector outputs and pick the best threshold.

Input CSV must contain at least:
  - label  (0 = human, 1 = AI)     # if labels are strings, they are normalized
  - p_ai   (probability of AI)     # float in [0,1]
Optional columns:
  - id, text, p_human, pred

Usage:
  python threshold_sweep.py results.csv \
      --metric f1 \
      --range 0.05 0.95 0.05 \
      --out_sweep sweep.csv \
      --out_preds preds_at_best.csv

Metrics supported for selection:
  - f1       (default)
  - acc      (accuracy)
  - youden   (TPR - FPR)

"""

import argparse
import math
import sys
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

def normalize_labels(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "human": "0", "ai": "1", "bot": "1", "machine": "1", "model": "1",
        "0": "0", "1": "1", "true": "1", "false": "0"
    }
    s = s.map(lambda x: mapping.get(x, x))
    out = pd.to_numeric(s, errors="coerce").fillna(0).astype(int).values
    return np.clip(out, 0, 1)

def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int,int,int,int]:
    # Returns TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)

def youden_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Youden's J = TPR - FPR
    tn, fp, fn, tp = compute_confusion(y_true, y_pred)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr - fpr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=str, help="Path to results CSV (with columns label, p_ai).")
    ap.add_argument("--metric", type=str, default="f1", choices=["f1","acc","youden"],
                    help="Selection metric for best threshold.")
    ap.add_argument("--range", type=float, nargs=3, default=[0.05, 0.95, 0.05],
                    metavar=("START","STOP","STEP"),
                    help="Threshold sweep: start stop step (inclusive start, inclusive if lands on stop).")
    ap.add_argument("--out_sweep", type=str, default=None,
                    help="Optional path to save the sweep table as CSV.")
    ap.add_argument("--out_preds", type=str, default=None,
                    help="Optional path to save predictions at best threshold.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "p_ai" not in df.columns:
        print("ERROR: input CSV must contain 'p_ai' column.", file=sys.stderr)
        sys.exit(2)
    if "label" not in df.columns:
        print("ERROR: input CSV must contain 'label' column.", file=sys.stderr)
        sys.exit(2)

    y = normalize_labels(df["label"])
    p = pd.to_numeric(df["p_ai"], errors="coerce").fillna(0.0).clip(0.0, 1.0).values

    # Basic sanity
    n = len(y)
    pos_rate = y.mean()
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = float("nan")
    print(f"[info] rows={n}  pos_rate(AI)=~{pos_rate:.3f}  ROC-AUC={auc:.3f}")

    start, stop, step = args.range
    if step <= 0:
        print("ERROR: step must be positive.", file=sys.stderr)
        sys.exit(2)

    # Construct thresholds (avoid floating drift)
    k = int(round((stop - start) / step)) + 1
    ths = [round(start + i * step, 10) for i in range(k)]
    if ths[-1] > stop + 1e-9:
        ths = [t for t in ths if t <= stop + 1e-9]

    rows = []
    best = None  # (score, thr, dict_of_metrics)

    for t in ths:
        pred = (p >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        acc = accuracy_score(y, pred)
        tn, fp, fn, tp = compute_confusion(y, pred)
        j = youden_index(y, pred)

        row = {
            "threshold": t,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "accuracy": acc,
            "youden": j,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp
        }
        rows.append(row)

        sel = {"f1": f1, "acc": acc, "youden": j}[args.metric]
        # tie-breaker: prefer lower threshold if scores equal within 1e-9
        if best is None or (sel > best[0] + 1e-9) or (abs(sel - best[0]) <= 1e-9 and t < best[1]):
            best = (sel, t, row)

    sweep_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

    # Report
    sel_name = args.metric.upper()
    score, thr, met = best
    print("\n=== Best threshold ===")
    print(f"Metric: {sel_name}")
    print(f"Threshold: {thr:.3f}")
    print(f"Precision: {met['precision']:.3f}")
    print(f"Recall:    {met['recall']:.3f}")
    print(f"F1:        {met['f1']:.3f}")
    print(f"Accuracy:  {met['accuracy']:.3f}")
    print(f"Youden J:  {met['youden']:.3f}")
    print("Confusion matrix [[TN FP],[FN TP]]:")
    print(f"[[{met['tn']:6d} {met['fp']:6d}]\n [{met['fn']:6d} {met['tp']:6d}]]")

    # Save sweep table
    if args.out_sweep:
        sweep_df.to_csv(args.out_sweep, index=False)
        print(f"\n[info] Wrote sweep table → {args.out_sweep}")

    # Save predictions at best threshold (optional)
    if args.out_preds:
        pred_best = (p >= thr).astype(int)
        out_df = df.copy()
        out_df["pred"] = pred_best
        out_df["threshold_used"] = thr
        out_df.to_csv(args.out_preds, index=False)
        print(f"[info] Wrote predictions at best threshold → {args.out_preds}")

if __name__ == "__main__":
    main()
