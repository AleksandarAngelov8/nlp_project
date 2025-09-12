#!/usr/bin/env python3
# eval_raid_open_std.py — Ghostbuster-open on RAID (unified CSV)
# Output schema:
#   id,dataset,domain,detector,y_true,y_score,pred,threshold,text_len,meta

import os, csv, json, time, argparse, math
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import dill as pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Ghostbuster feature utils (from repo) ---
from utils.featurize import t_featurize_logprobs, score_ngram
from utils.symbolic import get_words, vec_functions, scalar_functions

# ================ Unified CSV Collector ================
class StdCollector:
    def __init__(self, dataset: str, domain: str, detector: str, split: str = "all",
                 random_state: int = 42, enforce_3k_per_class: bool = False):
        self.dataset = dataset
        self.domain = domain
        self.detector = detector
        self.split = split
        self.rng = np.random.default_rng(random_state)
        self.rows = []  # (idx, y_true, y_score, text_len, meta, pred, threshold)
        self.enforce = enforce_3k_per_class

    def add(self, idx: int, y_true: int, y_score: float, text_len: int,
            meta: Optional[dict] = None, pred: Optional[int] = None, threshold: Optional[float] = None):
        self.rows.append((int(idx), int(y_true), float(y_score), int(text_len), meta or {}, pred, threshold))

    def _balanced_indexes(self, n_per_class: int = 3000):
        ys = np.array([r[1] for r in self.rows])
        idxs = np.arange(len(self.rows))
        h = idxs[ys == 0]
        a = idxs[ys == 1]
        if len(h) < n_per_class or len(a) < n_per_class:
            raise ValueError(f"Not enough to sample {n_per_class}/class (human={len(h)}, ai={len(a)}).")
        sel_h = self.rng.choice(h, size=n_per_class, replace=False)
        sel_a = self.rng.choice(a, size=n_per_class, replace=False)
        sel = np.concatenate([sel_h, sel_a])
        self.rng.shuffle(sel)
        return sel

    def finalize(self, out_csv: str, known_flip: bool = False, threshold: Optional[float] = None):
        rows = self.rows
        if self.enforce:
            sel = self._balanced_indexes(3000)
            rows = [rows[i] for i in sel]

        outp = Path(out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id","dataset","domain","detector","y_true","y_score","pred","threshold","text_len","meta"])
            for (idx, y_true, y_score, text_len, meta, pred_val, thr_val) in rows:
                rid = f"{self.dataset}_{self.split}_{idx}"
                w.writerow([
                    rid,
                    self.dataset,
                    self.domain,
                    self.detector,
                    int(y_true),
                    f"{float(y_score):.6f}",
                    "" if pred_val is None else int(pred_val),
                    "" if thr_val  is None else f"{float(thr_val):.6f}",
                    int(text_len),
                    json.dumps(meta, ensure_ascii=False),
                ])
        return {"out": str(outp), "n_total": len(self.rows), "n_written": len(rows)}

# ================ CLI / args ================
def parse_args():
    ap = argparse.ArgumentParser(description="Ghostbuster-open on RAID with unified CSV output.")
    ap.add_argument("--csv", required=True, help="Path to RAID CSV (e.g., train.csv / extra.csv).")
    ap.add_argument("--text_col", default="generation",
                    help="Text column; fallbacks: response,text,output,content.")
    ap.add_argument("--label_col", default=None,
                    help="Label column (0/1 or human/ai). If omitted, derive from 'model'.")
    ap.add_argument("--assume_label", type=int, default=None,
                    help="Force all labels to 0(human)/1(ai), override others.")
    ap.add_argument("--id_col", default="id", help="Row ID column (if missing, auto-generate).")
    ap.add_argument("--limit", type=int, default=-1, help="Global row cap (after shuffle). -1 = all.")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle before limiting.")
    ap.add_argument("--enforce_3k", action="store_true", help="Presample to 3k/3k BEFORE scoring.")
    ap.add_argument("--strict_fail", action="store_true",
                    help="With --enforce_3k, error if <3k per class after filters.")
    ap.add_argument("--domain_tag", default="raid-custom",
                    help='CSV domain tag. E.g., "raid-train" / "raid-extra".')

    ap.add_argument("--model", default=os.getenv("GB_OPEN_LM", "gpt2-medium"),
                    help="Open LM for surrogate logprobs (e.g., gpt2, gpt2-medium).")
    ap.add_argument("--max_ctx", type=int, default=int(os.getenv("GB_OPEN_MAX_CTX", "768")),
                    help="Max tokens fed to open LM.")
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"], help="Force device.")
    ap.add_argument("--threshold", type=float, default=-1.0,
                    help="If >=0, also fill pred/threshold.")
    ap.add_argument("--out", default="outputs/unified/gbopen_raid.csv",
                    help="Unified CSV output path.")
    return ap.parse_args()

# ================ Utils ================
def _fmt_eta(secs: float | None) -> str:
    if not secs or secs < 0 or math.isinf(secs) or math.isnan(secs): return "--:--:--"
    secs = int(secs + 0.5)
    h, r = divmod(secs, 3600); m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

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

# ================ GB-open feature pipeline ================
@torch.inference_mode()
def token_probs_and_subwords(text: str, tok, mdl, max_ctx: int):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_ctx)
    # 🔧 enforce integer dtypes for CUDA embeddings
    input_ids = enc["input_ids"].to(mdl.device, dtype=torch.long)
    attn      = enc["attention_mask"].to(mdl.device, dtype=torch.long)
    if input_ids.shape[1] < 2:
        eos_id = tok.eos_token_id if tok.eos_token_id is not None else (tok.pad_token_id or 0)
        input_ids = torch.cat([input_ids, torch.tensor([[eos_id]], device=mdl.device, dtype=torch.long)], dim=1)
        attn      = torch.cat([attn, torch.ones_like(attn, dtype=torch.long)], dim=1)
    out = mdl(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logits = out.logits
    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    next_ids = input_ids[:, 1:].unsqueeze(-1)
    tok_logp = torch.gather(logprobs, dim=-1, index=next_ids).squeeze(0).squeeze(-1)
    probs = torch.exp(tok_logp).detach().cpu().numpy()
    subwords = tok.convert_ids_to_tokens(input_ids[0, 1:].tolist())
    return probs, subwords

def build_feature_vector(text: str, best_features: List[str], trigram_model, enc_for_trigram,
                         tok, mdl, max_ctx: int) -> np.ndarray:
    probs, subwords = token_probs_and_subwords(text, tok, mdl, max_ctx)
    trigram_arr = np.array(score_ngram(text, trigram_model,      enc_for_trigram, n=3, strip_first=False))
    unigram_arr = np.array(score_ngram(text, trigram_model.base, enc_for_trigram, n=1, strip_first=False))
    t_features = t_featurize_logprobs(probs, probs, subwords)
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
                nxt = vector_map[tokens[i+1]]; curr = vec_functions[tk](curr, nxt); i += 2
            elif tk in scalar_functions:
                exp_features.append(float(scalar_functions[tk](curr))); break
            else:
                break
    return np.array(t_features + exp_features, dtype=float)

# ================ Main ================
def main():
    args = parse_args()

    # Load GB artifacts
    MODEL = pickle.load(open("model/model", "rb"))
    MU    = pickle.load(open("model/mu", "rb"))
    SIGMA = pickle.load(open("model/sigma", "rb"))
    best_features = Path("model/features.txt").read_text(encoding="utf-8").strip().split("\n")

    # Trigram + safe encoder
    import tiktoken
    trigram_model = pickle.load(open("model/trigram_model.pkl", "rb"))
    enc_for_trigram = tiktoken.encoding_for_model("davinci").encode

    # Open-LM and device
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

    # Load RAID CSV
    print(f"[info] Loading CSV: {args.csv}")
    try:
        df = pd.read_csv(args.csv, engine="pyarrow")
    except Exception:
        df = pd.read_csv(args.csv)

    # Column resolution
    text_col = args.text_col
    if text_col not in df.columns:
        for c in ["response", "text", "output", "content"]:
            if c in df.columns:
                print(f"[warn] text_col '{args.text_col}' not found; using '{c}'")
                text_col = c
                break
        else:
            raise ValueError(f"Text column not found. Tried '{args.text_col}', fallbacks ['response','text','output','content'].")

    # Labels
    if args.assume_label is not None:
        y_true = np.full(len(df), int(args.assume_label), dtype=int)
        print(f"[info] Forcing all labels to {args.assume_label} via --assume_label.")
    elif args.label_col and args.label_col in df.columns:
        y_true = normalize_labels(df[args.label_col])
        print(f"[info] Using explicit label_col='{args.label_col}'.")
    elif "label" in df.columns:
        y_true = normalize_labels(df["label"]); print("[info] Using 'label' column.")
    elif "is_human" in df.columns:
        y_true = normalize_labels(df["is_human"]); print("[info] Using 'is_human' column.")
    elif "model" in df.columns:
        y_true = derive_labels_from_model(df["model"]); print("[info] Derived labels from 'model' (non-empty => AI=1).")
    else:
        raise ValueError("No labels available. Provide --label_col or --assume_label, or ensure 'model' exists.")

    # Optional shuffle/limit
    if args.shuffle:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True); y_true = y_true[df.index]
    if args.limit and args.limit > 0 and args.limit < len(df):
        df = df.iloc[:args.limit].reset_index(drop=True); y_true = y_true[:len(df)]
    print(f"[info] Rows after limit/shuffle: {len(df)}")

    # Optional per-class presampling to 3k/3k BEFORE scoring
    if args.enforce_3k:
        idx_h = np.where(y_true == 0)[0]
        idx_a = np.where(y_true == 1)[0]
        if len(idx_h) < 3000 or len(idx_a) < 3000:
            msg = f"[WARN] enforce_3k: insufficient (human={len(idx_h)}, ai={len(idx_a)})"
            if args.strict_fail: raise RuntimeError(msg)
            print(msg, "Proceeding with as-many-as-available.")
        else:
            rng = np.random.default_rng(42)
            sel_h = rng.choice(idx_h, size=3000, replace=False)
            sel_a = rng.choice(idx_a, size=3000, replace=False)
            sel = np.concatenate([sel_h, sel_a]); rng.shuffle(sel)
            df = df.iloc[sel].reset_index(drop=True)
            y_true = y_true[sel]
            print("[info] Presampled to exactly 3k/3k prior to inference.")

    # Prepare collector
    detector = f"ghostbuster-open-{args.model.replace('-', '')}"
    collector = StdCollector(
        dataset="raid",
        domain=args.domain_tag,
        detector=detector,
        split="custom",
        random_state=42,
        enforce_3k_per_class=False  # presampling already applied if requested
    )

    # Threshold?
    use_thr = (args.threshold is not None) and (args.threshold >= 0.0)
    thr_val = args.threshold if use_thr else None

    # Which meta fields to retain in JSON
    meta_cols = ["domain", "attack", "model", "decoding", "repetition_penalty", "source_id", "adv_source_id"]

    # Scoring loop with ETA
    n = len(df)
    pbar = tqdm(total=n, desc="GB-open on RAID", unit="doc")
    ema_row_sec, alpha = None, 0.2
    for i, row in enumerate(df.itertuples(index=False)):
        t0 = time.perf_counter()
        txt = str(getattr(row, text_col, "") or "").strip()
        if txt:
            feats = build_feature_vector(txt, best_features, trigram_model, enc_for_trigram, tok, mdl, args.max_ctx)
            yscore = float(MODEL.predict_proba(((feats - MU) / SIGMA).reshape(1, -1))[:, 1][0])  # P(ai)
        else:
            yscore = 0.0

        # meta JSON (only fields that exist in CSV)
        meta = {}
        for c in meta_cols:
            if hasattr(row, c):
                val = getattr(row, c)
                if pd.isna(val): val = ""
                meta[c] = val
        # also keep original id if present / different
        if args.id_col in df.columns:
            meta["raid_id"] = str(getattr(row, args.id_col, ""))

        pred_val = int(yscore >= thr_val) if use_thr else None

        collector.add(
            idx=i,
            y_true=int(y_true[i]),
            y_score=yscore,
            text_len=len(txt),
            meta=meta,
            pred=pred_val,
            threshold=thr_val
        )

        dt = time.perf_counter() - t0
        ema_row_sec = dt if ema_row_sec is None else (alpha*dt + (1-alpha)*ema_row_sec)
        left = n - (i+1)
        eta = left * (ema_row_sec or 0.0)
        pbar.set_postfix({
            "r/s": f"{(1.0/(ema_row_sec or 1e-9)):.2f}",
            "avg_s/row": f"{(ema_row_sec or 0):.3f}",
            "ETA": _fmt_eta(eta),
            "fin": (datetime.now() + timedelta(seconds=eta)).strftime("%H:%M:%S") if eta>0 else "--:--:--"
        })
        pbar.update(1)
    pbar.close()

    info = collector.finalize(out_csv=args.out, known_flip=False, threshold=thr_val)
    print(f"[DONE] Unified CSV: {info['out']}  collected={info['n_total']}  written={info['n_written']}")

if __name__ == "__main__":
    main()
