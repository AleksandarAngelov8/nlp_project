# save as eval_csv.py ; run: python eval_csv.py <path_to_csv> <detector_name>
import sys, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

csv_path = sys.argv[1]
name = sys.argv[2] if len(sys.argv)>2 else "detector"
df = pd.read_csv(csv_path)
y_true = df["label"].astype(int).values

if "p_ai" in df.columns:
    scores = df["p_ai"].astype(float).values
elif "score" in df.columns:
    scores = df["score"].astype(float).values
else:
    # fall back to hard predictions only
    preds = df["pred"].astype(int).values
    print(f"[{name}] ACC={accuracy_score(y_true,preds):.3f}  F1={f1_score(y_true,preds):.3f}")
    print(confusion_matrix(y_true,preds))
    sys.exit(0)

# default hard label at 0.5 for proba, at 0 for binoculars score
import numpy as np
if "p_ai" in df.columns:
    preds = (scores >= 0.5).astype(int)
else:
    preds = (scores > 0.0).astype(int)

print(f"[{name}] ACC={accuracy_score(y_true,preds):.3f}  F1={f1_score(y_true,preds):.3f}")
try:
    # convert raw scores to a 0..1 direction where higher => AI
    if "p_ai" in df.columns:
        auc = roc_auc_score(y_true, scores)
    else:
        # binoculars score already "higher => AI"
        auc = roc_auc_score(y_true, scores)
    print(f"[{name}] ROC-AUC={auc:.3f}")
except Exception:
    pass
print(confusion_matrix(y_true,preds))
