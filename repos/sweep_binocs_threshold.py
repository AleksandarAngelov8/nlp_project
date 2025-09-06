#!/usr/bin/env python3
import argparse, json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--metric", default="f1", choices=["f1","accuracy","youden"])
ap.add_argument("--out_json", default=None, help="Optional file to save {'threshold': t}")
args = ap.parse_args()

df = pd.read_csv(args.csv)
# Coerce columns
if "label" not in df.columns or "score" not in df.columns:
    # infer common names
    label_col = "label" if "label" in df.columns else ("y_true" if "y_true" in df.columns else None)
    score_col = "score" if "score" in df.columns else ("y_score" if "y_score" in df.columns else None)
    if not label_col or not score_col:
        raise SystemExit("CSV must have 'label' and 'score' (or recognizable aliases).")
    df = df.rename(columns={label_col: "label", score_col: "score"})

y = df["label"].astype(int).to_numpy()
s = df["score"].astype(float).to_numpy()

# Optional: show ranking metrics (threshold-free)
try:
    roc_auc = roc_auc_score(y, s)
    pr_auc  = average_precision_score(y, s)
    print(f"ROC AUC={roc_auc:.4f}  PR AUC={pr_auc:.4f}")
except Exception:
    pass

# Build a dense threshold grid around observed scores
scores_sorted = np.unique(np.clip(s, -1e9, 1e9))
if len(scores_sorted) > 2000:
    # subsample to speed up
    idx = np.linspace(0, len(scores_sorted)-1, 2000, dtype=int)
    thr_grid = scores_sorted[idx]
else:
    thr_grid = scores_sorted

best = {"thr": None, "val": -1.0, "acc": None, "prec": None, "rec": None, "f1": None}
for t in thr_grid:
    yhat = (s >= t).astype(int)
    acc  = accuracy_score(y, yhat)
    prec = precision_score(y, yhat, zero_division=0)
    rec  = recall_score(y, yhat, zero_division=0)
    f1   = f1_score(y, yhat, zero_division=0)
    if args.metric == "f1":
        score = f1
    elif args.metric == "accuracy":
        score = acc
    else:
        # Youden's J (TPR + TNR - 1)
        tn = ((y==0) & (yhat==0)).sum()
        fp = ((y==0) & (yhat==1)).sum()
        fn = ((y==1) & (yhat==0)).sum()
        tp = ((y==1) & (yhat==1)).sum()
        tpr = tp/(tp+fn) if (tp+fn)>0 else 0
        tnr = tn/(tn+fp) if (tn+fp)>0 else 0
        score = tpr + tnr - 1
    if score > best["val"]:
        best = {"thr": float(t), "val": float(score), "acc": float(acc),
                "prec": float(prec), "rec": float(rec), "f1": float(f1)}

print("\n=== Best threshold ===")
print(f"threshold={best['thr']:.6f}")
print(f"metric={args.metric} -> {best['val']:.6f}")
print(f"ACC={best['acc']:.4f}  P={best['prec']:.4f}  R={best['rec']:.4f}  F1={best['f1']:.4f}")

if args.out_json:
    import json
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"threshold": best["thr"]}, f, indent=2)
    print(f"Saved to {args.out_json}")
