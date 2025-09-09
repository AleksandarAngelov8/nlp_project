import pandas as pd
import numpy as np

df = pd.read_csv("outputs/raid/roberta_train5k.csv")  # <-- your file
y = df["label"].values.astype(int)        # 1 = AI, 0 = human
p = df["p_ai"].values                     # model’s AI probability

def f1_at(t):
    pred = (p >= t).astype(int)
    tp = ((pred==1)&(y==1)).sum()
    fp = ((pred==1)&(y==0)).sum()
    fn = ((pred==0)&(y==1)).sum()
    if tp==0: return 0.0
    prec = tp / (tp+fp)
    rec  = tp / (tp+fn)
    return 2*prec*rec/(prec+rec+1e-12)

ths = np.linspace(0.01, 0.99, 99)
scores = [(t, f1_at(t)) for t in ths]
best_t, best_f1 = max(scores, key=lambda x: x[1])
print(f"Best threshold by F1: {best_t:.3f}  (F1={best_f1:.3f})")
