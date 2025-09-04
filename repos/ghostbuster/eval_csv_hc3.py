# eval_csv_hc3.py
import argparse, os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, f1_score, accuracy_score, confusion_matrix
)
import matplotlib.pyplot as plt

def recall_at_fpr(target, fpr, tpr):
    mask = fpr <= target
    return float(np.max(tpr[mask])) if np.any(mask) else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV with id,domain,y_true,y_score,detector_name,text_len")
    ap.add_argument("--outdir", default="outputs/ghostbuster_open/metrics_hc3")
    ap.add_argument("--name", default="ghostbuster_open_hc3")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    y = df["y_true"].astype(int).values
    s = df["y_score"].astype(float).values

    # Core metrics
    roc_auc = roc_auc_score(y, s)
    pr_auc = average_precision_score(y, s)

    # @0.5
    y05 = (s >= 0.5).astype(int)
    f1_05 = f1_score(y, y05)
    acc_05 = accuracy_score(y, y05)
    cm_05 = confusion_matrix(y, y05).ravel()  # tn, fp, fn, tp

    # Best-F1 threshold
    uniq = np.unique(s)
    if len(uniq) > 2000:
        thrs = np.linspace(uniq.min(), uniq.max(), 2000)
    else:
        thrs = uniq
    best_f1, best_acc, best_thr = -1.0, None, None
    for t in thrs:
        yp = (s >= t).astype(int)
        f1 = f1_score(y, yp)
        if f1 > best_f1:
            best_f1, best_acc, best_thr = f1, accuracy_score(y, yp), float(t)
    cm_best = confusion_matrix(y, (s >= best_thr).astype(int)).ravel()

    # Fixed-FPR recalls (+ implied threshold for 5%)
    fpr, tpr, thr = roc_curve(y, s)
    rec_1 = recall_at_fpr(0.01, fpr, tpr)
    rec_5 = recall_at_fpr(0.05, fpr, tpr)
    idx_5 = np.where(fpr <= 0.05)[0]
    thr_5 = float(thr[idx_5[np.argmax(tpr[idx_5])]]) if len(idx_5) else None
    if thr_5 is not None:
        y5 = (s >= thr_5).astype(int)
        f1_5, acc_5 = f1_score(y, y5), accuracy_score(y, y5)
        cm_5 = confusion_matrix(y, y5).ravel()
    else:
        f1_5 = acc_5 = None
        cm_5 = [None]*4

    # Save metrics table
    metr = pd.DataFrame({
        "Metric": [
            "ROC AUC", "PR AUC (AP)",
            "F1 @ 0.5", "ACC @ 0.5",
            "Best F1", "ACC @ Best F1", "Best-F1 Threshold",
            "Recall @ 1% FPR", "Recall @ 5% FPR",
            "F1 @ 5% FPR", "ACC @ 5% FPR", "Threshold @ 5% FPR",
            "Rows"
        ],
        "Value": [
            round(roc_auc,4), round(pr_auc,4),
            round(f1_05,4), round(acc_05,4),
            round(best_f1,4), round(best_acc,4), best_thr,
            round(rec_1,4), round(rec_5,4),
            None if f1_5 is None else round(f1_5,4),
            None if acc_5 is None else round(acc_5,4),
            thr_5, len(df)
        ]
    })
    metr_path = os.path.join(args.outdir, f"{args.name}_summary.csv")
    metr.to_csv(metr_path, index=False)

    # Save confusion matrices
    def cm_df(tn, fp, fn, tp):
        return pd.DataFrame({"": ["Pred 0","Pred 1"],
                             "True 0":[tn, fp],"True 1":[fn, tp]}).set_index("")
    cms = {
        "cm_05.csv": cm_df(*cm_05),
        "cm_best.csv": cm_df(*cm_best),
        "cm_at_5fpr.csv": cm_df(*cm_5) if cm_5[0] is not None else None
    }
    for name, cdf in cms.items():
        if cdf is not None:
            cdf.to_csv(os.path.join(args.outdir, f"{args.name}_{name}"))

    # ROC plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC — Ghostbuster OOD on HC3")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{args.name}_roc.png"))
    plt.close()

    # PR plot
    from sklearn.metrics import precision_recall_curve
    prec, rec, _ = precision_recall_curve(y, s)
    plt.figure()
    plt.plot(rec, prec, label=f"AP={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall — Ghostbuster OOD on HC3")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{args.name}_pr.png"))
    plt.close()

    print(f"[OK] Wrote:\n  {metr_path}\n  (and confusion matrices + ROC/PR plots in {args.outdir})")

if __name__ == "__main__":
    main()
