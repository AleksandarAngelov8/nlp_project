# run_roberta_ghostbuster.py  (std schema, per-domain CSVs with graceful fallback)
import argparse, math, numpy as np, torch, tqdm, csv, os
from typing import List, Tuple
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch.nn import functional as F
from utils.load import get_generate_dataset, Dataset
from std_eval import StdCollector
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using CUDA..." if device.type == "cuda" else "Using CPU...")

roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Data roots (your absolute paths kept)
wp_dataset = [
    Dataset("normal", "C:/Users/kucem/PraktikumProject/nlp_project/repos/ghostbuster/data/wp/human"),
    Dataset("normal", "C:/Users/kucem/PraktikumProject/nlp_project/repos/ghostbuster/data/wp/gpt"),
]
reuter_dataset = [
    Dataset("author", "C:/Users/kucem/PraktikumProject/nlp_project/repos/ghostbuster/data/reuter/human"),
    Dataset("author", "C:/Users/kucem/PraktikumProject/nlp_project/repos/ghostbuster/data/reuter/gpt"),
]
essay_dataset = [
    Dataset("normal", "C:/Users/kucem/PraktikumProject/nlp_project/repos/ghostbuster/data/essay/human"),
    Dataset("normal", "C:/Users/kucem/PraktikumProject/nlp_project/repos/ghostbuster/data/essay/gpt"),
]

class RobertaDataset(TorchDataset):
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts, self.labels = texts, labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        enc = roberta_tokenizer(self.texts[idx], return_tensors="pt",
                                truncation=True, padding="max_length", max_length=512)
        return {"input_ids": enc["input_ids"].squeeze().to(device),
                "attention_mask": enc["attention_mask"].squeeze().to(device),
                "labels": torch.tensor(self.labels[idx]).to(device)}

def get_scores(labels, probs, calibrated=False, precision=6):
    thr = sorted(probs)[len(labels)-sum(labels)-1] if calibrated else 0.5
    if sum(labels)==0:
        return round(accuracy_score(labels, probs>thr),precision), round(f1_score(labels, probs>thr),precision), -1
    return (round(accuracy_score(labels, probs>thr),precision),
            round(f1_score(labels, probs>thr),precision),
            round(roc_auc_score(labels, probs),precision))

def train_roberta_model(train_text, train_labels, out_dir, max_epochs=1):
    mdl = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)
    opt = torch.optim.SGD(mdl.parameters(), lr=0.001); loss_fn = torch.nn.CrossEntropyLoss()
    idx = np.arange(len(train_text)); np.random.shuffle(idx)
    tr, va = idx[: math.floor(0.8*len(idx))], idx[math.floor(0.8*len(idx)) :]
    tr_ds = RobertaDataset([train_text[i] for i in tr], np.array(train_labels)[tr].tolist())
    va_ds = RobertaDataset([train_text[i] for i in va], np.array(train_labels)[va].tolist())
    tr_ld = DataLoader(tr_ds, batch_size=8, shuffle=True); va_ld = DataLoader(va_ds, batch_size=8, shuffle=True)
    best = float("inf")
    for ep in range(max_epochs):
        mdl.train()
        for batch in tqdm.tqdm(tr_ld, desc=f"Train epoch {ep+1}"):
            opt.zero_grad(); out = mdl(**batch); loss = loss_fn(out.logits, batch["labels"]); loss.backward(); opt.step()
        mdl.eval(); val_loss=0.0
        with torch.no_grad():
            for batch in tqdm.tqdm(va_ld, desc="Valid"):
                out = mdl(**batch); val_loss += loss_fn(out.logits, batch["labels"]).item()
        val_loss /= max(1,len(va_ld)); print(f"Epoch {ep+1} Validation Loss: {val_loss:.6f}")
        if val_loss>best: break
        best=val_loss; os.makedirs(out_dir, exist_ok=True); mdl.save_pretrained(out_dir)

def load_texts_labels(gen_fn, indices=None, filter_fn=lambda f: True) -> Tuple[List[str], List[int], List[str]]:
    files = gen_fn(lambda f: f)[indices] if indices is not None else gen_fn(lambda f: f)
    texts, labels, paths = [], [], []
    for fp in files:
        if not filter_fn(fp): continue
        with open(fp, "r", encoding="utf-8") as fh: txt = fh.read()
        texts.append(txt); labels.append(int("gpt" in fp)); paths.append(fp)
    return texts, labels, paths

def run_roberta_prob(model_name: str, texts: List[str], batch_size: int = 8) -> List[float]:
    local_dir = os.path.join("models", f"roberta_{model_name}")
    if os.path.isdir(local_dir):
        load_id = local_dir
    else:
        print(f"[WARN] Local model not found: {local_dir}. Falling back to Hello-SimpleAI/chatgpt-detector-roberta.")
        load_id = "Hello-SimpleAI/chatgpt-detector-roberta"
    mdl = RobertaForSequenceClassification.from_pretrained(load_id, num_labels=2).to(device)
    mdl.eval(); probs=[]
    for i in tqdm.tqdm(range(0,len(texts),batch_size), desc=f"Infer {model_name}"):
        batch = texts[i:i+batch_size]
        enc = roberta_tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
        enc = {k:v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            out = mdl(**enc); p = F.softmax(out.logits, dim=1)[:,1]
        probs.extend(p.detach().cpu().tolist())
    return probs

def write_unified_csv_for_domain(domain: str, model_name: str, gen_fn_all, out_csv: str,
                                 indices=None, enforce_3k=True, strict_fail=False):
    texts, labels, paths = load_texts_labels(gen_fn_all, indices=indices, filter_fn=lambda x: domain in x)
    print(f"[{domain}] loaded: {len(texts)} examples")
    # quick availability check before running inference
    n_h = sum(1 for y in labels if y==0); n_a = sum(1 for y in labels if y==1)
    need = 3000
    local_enforce = enforce_3k
    if enforce_3k and (n_h<need or n_a<need):
        msg = f"[{domain}] insufficient for 3k/3k (human={n_h}, ai={n_a})."
        if strict_fail:
            raise RuntimeError(msg + " Use --strict_fail False or collect more data.")
        print("[WARN]", msg, "Writing as-many-as-available for this domain.")
        local_enforce = False

    probs = run_roberta_prob(model_name=model_name, texts=texts, batch_size=8)

    collector = StdCollector(
        dataset="ghostbuster",
        domain=f"gb-{domain}",
        detector="roberta-gb",
        split=domain,
        random_state=42,
        enforce_3k_per_class=local_enforce,
    )
    for idx, (y,s,txt,pth) in enumerate(zip(labels, probs, texts, paths)):
        collector.add(idx=idx, y_true=int(y), y_score=float(s), text_len=len(txt),
                      meta={"path": pth, "seed_split": "test",
                            "model_dir": f"models/roberta_{model_name}",
                            "tokenizer":"roberta-base"})
    info = collector.finalize(out_csv=out_csv, known_flip=False, threshold=None)
    print(f"[{domain}] CSV: {info['out']}  n_total={info['n_total']}  n_written={info['n_written']}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="roberta_results")
    ap.add_argument("--enforce_3k", action="store_true", default=False,
                    help="Require 3000 per class per domain; otherwise write as many.")
    ap.add_argument("--strict_fail", action="store_true", default=False,
                    help="If --enforce_3k and a domain is short, hard-fail instead of falling back.")
    args = ap.parse_args()

    np.random.seed(args.seed)
    idx = np.arange(6000); np.random.shuffle(idx)
    train_idx, test_idx = idx[: math.floor(0.8*len(idx))], idx[math.floor(0.8*len(idx)) :]
    print("Train/Test Split:", train_idx, test_idx)

    gen_fn_all = get_generate_dataset(*wp_dataset, *reuter_dataset, *essay_dataset)

    # Optional training (unchanged)
    # ...

    if args.run:
        os.makedirs(args.out_dir, exist_ok=True)

        # Optional metrics (unchanged) — NOTE: use "reuter" now, not "only_reuter"
        # ...

        # Unified CSVs (one per domain)
        out_wp     = os.path.join(args.out_dir, "ghostbuster_roberta_wp.csv")
        out_reuter = os.path.join(args.out_dir, "ghostbuster_roberta_reuter.csv")
        out_essay  = os.path.join(args.out_dir, "ghostbuster_roberta_essay.csv")

        write_unified_csv_for_domain("wp",     "gpt",    gen_fn_all, out_wp,
                                     indices=test_idx, enforce_3k=args.enforce_3k, strict_fail=args.strict_fail)
        write_unified_csv_for_domain("reuter", "reuter", gen_fn_all, out_reuter,
                                     indices=test_idx, enforce_3k=args.enforce_3k, strict_fail=args.strict_fail)
        write_unified_csv_for_domain("essay",  "gpt",    gen_fn_all, out_essay,
                                     indices=test_idx, enforce_3k=args.enforce_3k, strict_fail=args.strict_fail)

        print("Done.")
