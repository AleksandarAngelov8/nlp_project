import sys, os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

def load_scores(csv_path):
    df = pd.read_csv(csv_path)
    # detect score column: binoculars -> "score", roberta -> "p_ai"
    score_col = None
    for c in ["score", "p_ai", "prob_ai", "proba_ai"]:
        if c in df.columns:
            score_col = c
            break
    if score_col is None:
        raise ValueError(f"No score column found in {csv_path}. Expected one of: score, p_ai, prob_ai, proba_ai")
    if "label" not in df.columns:
        raise ValueError(f"No 'label' column found in {csv_path}")

    y = df["label"].astype(int).to_numpy()
    s = pd.to_numeric(df[score_col], errors="coerce").to_numpy()

    # drop rows with NaN scores (if any)
    mask = ~np.isnan(s)
    if mask.sum() < len(s):
        print(f"[WARN] Dropping {len(s) - mask.sum()} rows with NaN scores in {os.path.basename(csv_path)}")
        y = y[mask]
        s = s[mask]
        df = df.loc[mask].reset_index(drop=True)
    return df, y, s, score_col

def find_threshold_at_fpr(y_true, scores, target_fpr=0.05):
    """
    Scan thresholds to find one whose FPR is closest to target_fpr.
    Returns (threshold, actual_fpr).
    """
    # Ensure higher score => more likely AI. If your detector is reversed, flip here.
    uniq = np.unique(scores)
    # For efficiency on very large sets, you can subsample thresholds:
    # uniq = np.quantile(scores, np.linspace(0,1,2001))

    best_thr = None
    best_diff = float("inf")
    best_fpr = None

    # Compute TN/FP efficiently for many thresholds by sorting
    order = np.argsort(scores)
    s_sorted = scores[order]
    y_sorted = y_true[order]

    # cumulative counts for negatives (y=0) and positives (y=1)
    neg = (y_sorted == 0).astype(int)
    pos = 1 - neg
    c_neg = np.cumsum(neg)                # up to idx inclusive
    c_pos = np.cumsum(pos)

    total_neg = c_neg[-1]
    total_pos = c_pos[-1]

    # For threshold t, predict AI when score >= t.
    # Using sorted scores, idx = first position where s_sorted[idx] >= t
    # Then FP = number of negatives in [idx .. end], TN = c_neg[idx-1]
    # We'll scan over unique values as candidate thresholds.
    for t in uniq:
        idx = np.searchsorted(s_sorted, t, side="left")
        fp = total_neg - (c_neg[idx-1] if idx > 0 else 0)
        tn = (c_neg[idx-1] if idx > 0 else 0)
        fpr = fp / max(total_neg, 1)

        diff = abs(fpr - target_fpr)
        if diff < best_diff:
            best_diff = diff
            best_thr = t
            best_fpr = fpr

    return float(best_thr), float(best_fpr)

def evaluate_with_threshold(y_true, scores, thr):
    preds = (scores >= thr).astype(int)
    acc = accuracy_score(y_true, preds)
    f1  = f1_score(y_true, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_true, preds, labels=[0,1])
    return acc, f1, auc, cm, preds

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python calibrate_fixed_fpr.py <cal_csv> <name> [target_fpr=0.05] [eval_csv=] [out_calibrated_csv=]")
        sys.exit(1)

    cal_csv = sys.argv[1]
    name    = sys.argv[2]
    target_fpr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    eval_csv = sys.argv[4] if len(sys.argv) > 4 else ""
    out_csv  = sys.argv[5] if len(sys.argv) > 5 else ""

    # Load calibration set
    cal_df, cal_y, cal_s, score_col = load_scores(cal_csv)
    thr, actual_fpr = find_threshold_at_fpr(cal_y, cal_s, target_fpr=target_fpr)
    cal_acc, cal_f1, cal_auc, cal_cm, cal_pred = evaluate_with_threshold(cal_y, cal_s, thr)

    print(f"\n[{name}] Calibration on {os.path.basename(cal_csv)}")
    print(f"Target FPR: {target_fpr:.3f}  →  Threshold: {thr:.6f}  (Actual FPR: {actual_fpr:.3f})")
    print(f"ACC={cal_acc:.3f}  F1={cal_f1:.3f}  ROC-AUC={cal_auc:.3f}")
    print(cal_cm)

    # Optionally save calibrated predictions for the calibration set
    if out_csv and not eval_csv:
        cal_out = cal_df.copy()
        cal_out["threshold"] = thr
        cal_out["pred_calibrated"] = cal_pred
        cal_out.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"[Saved] {out_csv}")

    # Evaluate on a separate eval set (preferred)
    if eval_csv:
        eval_df, eval_y, eval_s, _ = load_scores(eval_csv)
        ev_acc, ev_f1, ev_auc, ev_cm, ev_pred = evaluate_with_threshold(eval_y, eval_s, thr)

        print(f"\n[{name}] Evaluation on {os.path.basename(eval_csv)} (using calibrated thr)")
        print(f"ACC={ev_acc:.3f}  F1={ev_f1:.3f}  ROC-AUC={ev_auc:.3f}")
        print(ev_cm)

        if out_csv:
            ev_out = eval_df.copy()
            ev_out["threshold"] = thr
            ev_out["pred_calibrated"] = ev_pred
            ev_out.to_csv(out_csv, index=False, encoding="utf-8")
            print(f"[Saved] {out_csv}")

if __name__ == "__main__":
    main()
