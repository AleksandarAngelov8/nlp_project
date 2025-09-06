#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--in_csv", required=True)
ap.add_argument("--out_csv", required=True)
ap.add_argument("--threshold", type=float, required=True)
args = ap.parse_args()

df = pd.read_csv(args.in_csv)
# normalize columns
if "score" not in df.columns:
    raise SystemExit("CSV must contain a 'score' column.")
df["pred"] = np.where(df["score"].astype(float) >= args.threshold, 1, 0)
df.to_csv(args.out_csv, index=False, encoding="utf-8")
print(f"Rewrote predictions with threshold={args.threshold} -> {args.out_csv}")
