#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Ghostbuster-open on Ghostbuster datasets (wp, reuter, essay)
# Produces THREE unified CSVs, one per domain:
#   <out_dir>/ghostbuster_open_wp.csv
#   <out_dir>/ghostbuster_open_reuter.csv
#   <out_dir>/ghostbuster_open_essay.csv
#
# Schema: id,dataset,domain,detector,y_true,y_score,pred,threshold,text_len,meta
# - dataset="ghostbuster"
# - domain in {"gb-wp","gb-reuter","gb-essay"}
# - detector="ghostbuster-open-<model>"  e.g., "ghostbuster-open-gpt2medium"
# - y_true: 0=human, 1=ai (derived from file path containing "gpt")
# - y_score: P(ai) from Ghostbuster-open classifier (higher ⇒ AI)
# - pred/threshold blank unless --threshold >= 0
# - Enforces 3k/3k per domain BEFORE inference by default (fast). Disable with --no_enforce_3k.

import os, csv, json, time, argparse, math, random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import dill as pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Repo utilities
from utils.load import get_generate_dataset, Dataset

# GB-open feature stack
from utils.featurize import t_featurize_logprobs, score_ngram
from utils.symbolic import get_words, vec_functions, scalar_functions

# -------------------- unified CSV collector --------------------
class StdCollector:
    def __init__(self, dataset, domain, detector, split="all", random_state=42, enforce_3k_per_class=False):
        self.dataset = dataset
        self.domain = domain
        self.detector = detector
        self.split = split
        self.rng = np.random.default_rng(random_state)
        self.rows = []  # (idx, y_true, y_score, text_len, meta, pred, threshold)
        self.enforce = enforce_3k_per_class

    def add(self, idx, y_true, y_score, text_len, meta=None, pred=None, threshold=None):
        self.rows.append((int(idx), int(y_true), float(y_score), int(text_len), meta or {}, pred, threshold))

    def _balanced_indexes(self, n_per_class=3000):
        ys = np.array([r[1] for r in self.rows])
        idxs = np.arange(len(self.rows))
        h = idxs[ys == 0]
        a = idxs[ys == 1]
        if len(h) < n_per_class or len(a) < n_per_class:
            raise ValueError(f"Not enough examples to sample {n_per_class} per class (human={len(h)}, ai={len(a)}).")
        sel_h = self.rng.choice(h, size=n_per_class, replace=False)
        sel_a = self.rng.choice(a, size=n_per_class, replace=False)
        sel = np.concatenate([sel_h, sel_a]); self.rng.shuffle(sel)
        return sel

    def finalize(self, out_csv, known_flip=False, threshold=None):
        rows = self.rows
        if self.enforce:
            sel = self._balanced_indexes(3000)
            rows = [rows[i] for i in sel]

        out = Path(out_csv); out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id","dataset","domain","detector","y_true","y_score","pred","threshold","text_len","meta"])
            for (idx, y_true, y_score, text_len, meta, pred_val, thr_val) in rows:
                rid = f"{self.dataset}_{self.split}_{idx}"
                w.writerow([
                    rid, self.dataset, self.domain, self.detector,
                    int(y_true), f"{y_score:.6f}",
                    "" if pred_val is None else int(pred_val),
                    "" if thr_val  is None else f"{float(thr_val):.6f}",
                    int(text_len), json.dumps(meta, ensure_ascii=False)
                ])
        return {"out": str(out), "n_total": len(self.rows), "n_written": len(rows), "flipped": False}

# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs/unified/gbopen_on_gb",
                    help="Directory to write the three unified CSVs.")
    ap.add_argument("--model", default=os.getenv("GB_OPEN_LM", "gpt2-medium"),
                    help="HF causal LM for surrogate logprobs (e.g., gpt2, gpt2-medium).")
    ap.add_argument("--max_ctx", type=int, default=int(os.getenv("GB_OPEN_MAX_CTX", "768")),
                    help="Max tokens fed into open LM.")
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"],
                    help="Force device; default auto.")
    ap.add_argument("--threshold", type=float, default=-1.0,
                    help="If >=0, also fill pred/threshold columns.")
    ap.add_argument("--no_enforce_3k", action="store_true",
                    help="Do NOT enforce per-domain 3k/3k (write as-many-as-available).")
    ap.add_argument("--strict_fail", action="store_true",
                    help="If enforcing 3k/3k and insufficient data, hard fail.")
    ap.add_argument("--seed", type=int, default=42)
    # Optional overrides for data roots; by default we resolve repo-relative paths.
    ap.add_argument("--data_root", default=None, help="Repo root; if omitted, auto-detect via script location.")
    ap.add_argument("--wp_dir", default=None, help="Override path to data/wp (folder with human/ and gpt/)")
    ap.add_argument("--reuter_dir", default=None, help="Override path to data/reuter (folder with human/ and gpt/)")
    ap.add_argument("--essay_dir", default=None, help="Override path to data/essay (folder with human/ and gpt/)")
    return ap.parse_args()

# -------------------- GB-open feature pipeline --------------------
@torch.inference_mode()
def token_probs_and_subwords(text, tok, mdl, max_ctx: int):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_ctx)

    # 🔧 Force integer types when moving to device
    input_ids = enc["input_ids"].to(mdl.device, dtype=torch.long)
    attn      = enc["attention_mask"].to(mdl.device, dtype=torch.long)

    # Ensure at least 2 tokens to align next-token probabilities
    if input_ids.shape[1] < 2:
        eos_id = tok.eos_token_id if tok.eos_token_id is not None else (tok.pad_token_id or 0)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[eos_id]], device=mdl.device, dtype=torch.long)],
            dim=1
        )
        attn = torch.cat([attn, torch.ones_like(attn, dtype=torch.long)], dim=1)

    out = mdl(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logits = out.logits
    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    next_ids = input_ids[:, 1:].unsqueeze(-1)
    tok_logp = torch.gather(logprobs, dim=-1, index=next_ids).squeeze(0).squeeze(-1)
    probs = torch.exp(tok_logp).cpu().numpy()
    subwords = tok.convert_ids_to_tokens(input_ids[0, 1:].tolist())
    return probs, subwords

def build_feature_vector(text, best_features, trigram_model, enc_for_trigram, tok, mdl, max_ctx: int):
    probs, subwords = token_probs_and_subwords(text, tok, mdl, max_ctx)
    trigram_arr = np.array(score_ngram(text, trigram_model,      enc_for_trigram, n=3, strip_first=False))
    unigram_arr = np.array(score_ngram(text, trigram_model.base, enc_for_trigram, n=1, strip_first=False))
    t_features = t_featurize_logprobs(probs, probs, subwords)  # alias both streams to open-LM probs

    vector_map = {
        "davinci-logprobs": probs,
        "ada-logprobs":     probs,
        "trigram-logprobs": trigram_arr,
        "unigram-logprobs": unigram_arr,
    }
    exp_features = []
    for exp in best_features:
        tokens = get_words(exp); curr = vector_map[tokens[0]]; i = 1
        while i < len(tokens):
            tk = tokens[i]
            if tk in vec_functions:
                nxt = vector_map[tokens[i+1]]; curr = vec_functions[tk](curr, nxt); i += 2
            elif tk in scalar_functions:
                exp_features.append(float(scalar_functions[tk](curr))); break
            else:
                break
    return np.array(t_features + exp_features, dtype=float)

# -------------------- helpers --------------------
def _detector_tag(model_name: str) -> str:
    return model_name.replace("-", "")

def _domain_tag(name: str) -> str:
    return {"wp":"gb-wp","reuter":"gb-reuter","essay":"gb-essay"}[name]

def _fmt_eta(secs: float | None) -> str:
    if not secs or secs < 0 or math.isinf(secs) or math.isnan(secs): return "--:--:--"
    secs = int(secs + 0.5); h, r = divmod(secs, 3600); m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def load_texts_labels_paths(gen_fn, indices=None, filter_fn=lambda f: True) -> Tuple[List[str], List[int], List[str]]:
    if indices is not None:
        files = gen_fn(lambda f: f)[indices]
    else:
        files = gen_fn(lambda f: f)
    texts, labels, paths = [], [], []
    for file in files:
        if not filter_fn(file):
            continue
        with open(file, "r", encoding="utf-8") as fh:
            txt = fh.read()
        texts.append(txt)
        labels.append(int("gpt" in file))
        paths.append(file)
    return texts, labels, paths

# -------------------- main --------------------
def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    # Resolve repo-relative data roots unless overridden
    if args.data_root:
        repo_root = Path(args.data_root)
    else:
        repo_root = Path(__file__).resolve().parent

    wp_root     = Path(args.wp_dir     or repo_root / "data" / "wp")
    reuter_root = Path(args.reuter_dir or repo_root / "data" / "reuter")
    essay_root  = Path(args.essay_dir  or repo_root / "data" / "essay")

    # Build dataset objects (these mirror what you used for RoBERTa)
    wp_dataset = [
        Dataset("normal", str(wp_root / "human")),
        Dataset("normal", str(wp_root / "gpt")),
    ]
    reuter_dataset = [
        Dataset("author", str(reuter_root / "human")),
        Dataset("author", str(reuter_root / "gpt")),
    ]
    essay_dataset = [
        Dataset("normal", str(essay_root / "human")),
        Dataset("normal", str(essay_root / "gpt")),
    ]

    # Generators
    gen_fn_all   = get_generate_dataset(*wp_dataset, *reuter_dataset, *essay_dataset)

    # Load GB-open classifier + normalization + features (same files you used for HC3/M4 runs)
    MODEL = pickle.load(open("model/model", "rb"))
    MU    = pickle.load(open("model/mu", "rb"))
    SIGMA = pickle.load(open("model/sigma", "rb"))
    best_features = Path("model/features.txt").read_text(encoding="utf-8").strip().split("\n")

    # Trigram model + safe encoder (tiktoken)
    import tiktoken
    trigram_model = pickle.load(open("model/trigram_model.pkl", "rb"))
    _enc = tiktoken.encoding_for_model("davinci")
    def safe_trigram_encode(s: str):
        return _enc.encode(s, disallowed_special=())
    enc_for_trigram = safe_trigram_encode

    # Device & open-LM
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Device: {device} | OPEN_LM={args.model} | MAX_CTX={args.max_ctx}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    try:
        if device == "cuda": torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Threshold usage?
    use_thr = (args.threshold is not None) and (args.threshold >= 0.0)
    thr_val = args.threshold if use_thr else None

    # Output directory
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    outs = {
        "wp":     out_dir / "ghostbuster_open_wp.csv",
        "reuter": out_dir / "ghostbuster_open_reuter.csv",
        "essay":  out_dir / "ghostbuster_open_essay.csv",
    }

    # The three domains
    domains = ["wp", "reuter", "essay"]

    # For each domain, presample to 3k/3k BEFORE inference (fast) unless disabled.
    need = 3000
    detector = f"ghostbuster-open-{(args.model.replace('-', ''))}"

    for domain in domains:
        print(f"\n==== Domain: {domain} ====")
        # Load domain subset
        texts, labels, paths = load_texts_labels_paths(gen_fn_all, indices=None, filter_fn=lambda x, d=domain: d in x)
        n_h = sum(1 for y in labels if y == 0); n_a = sum(1 for y in labels if y == 1)
        print(f"[info] loaded: {len(texts)} (human={n_h}, ai={n_a})")

        # Presample to 3k/3k per domain
        enforce = not args.no_enforce_3k
        if enforce:
            if n_h < need or n_a < need:
                msg = f"[WARN] {domain}: not enough for 3k/3k (human={n_h}, ai={n_a})."
                if args.strict_fail: raise RuntimeError(msg)
                print(msg, "Proceeding with as-many-as-available.")
            else:
                # indices of each class
                idx_h = [i for i,y in enumerate(labels) if y == 0]
                idx_a = [i for i,y in enumerate(labels) if y == 1]
                rng = np.random.default_rng(args.seed)
                sel_h = rng.choice(idx_h, size=need, replace=False)
                sel_a = rng.choice(idx_a, size=need, replace=False)
                sel = np.concatenate([sel_h, sel_a]); rng.shuffle(sel)
                texts  = [texts[i]  for i in sel]
                labels = [labels[i] for i in sel]
                paths  = [paths[i]  for i in sel]
                n_h, n_a = need, need
                print("[info] presampled to 3k/3k.")

        print(f"[info] scoring {len(texts)} rows…")

        # Collector
        collector = StdCollector(
            dataset="ghostbuster",
            domain=_domain_tag(domain),
            detector=detector,
            split=domain,
            random_state=args.seed,
            enforce_3k_per_class=False  # we presampled already if enforcing
        )

        # Scoring loop (with ETA)
        ema_row_sec, alpha = None, 0.2
        pbar = tqdm(total=len(texts), desc=f"GB-open on {domain}", unit="doc")
        for i, (txt, y, pth) in enumerate(zip(texts, labels, paths)):
            t0 = time.perf_counter()
            txt = (txt or "").strip()
            # Build features
            probs, subwords = token_probs_and_subwords(txt, tok, mdl, args.max_ctx)
            trigram_arr = np.array(score_ngram(txt, trigram_model,      enc_for_trigram, n=3, strip_first=False))
            unigram_arr = np.array(score_ngram(txt, trigram_model.base, enc_for_trigram, n=1, strip_first=False))
            t_features = t_featurize_logprobs(probs, probs, subwords)
            # Expand symbolic features
            vector_map = {
                "davinci-logprobs": probs,
                "ada-logprobs":     probs,
                "trigram-logprobs": trigram_arr,
                "unigram-logprobs": unigram_arr,
            }
            exp_features = []
            for exp in best_features:
                tokens = get_words(exp); curr = vector_map[tokens[0]]; j = 1
                while j < len(tokens):
                    tk = tokens[j]
                    if tk in vec_functions:
                        nxt = vector_map[tokens[j+1]]; curr = vec_functions[tk](curr, nxt); j += 2
                    elif tk in scalar_functions:
                        exp_features.append(float(scalar_functions[tk](curr))); break
                    else:
                        break
            feats = np.array(t_features + exp_features, dtype=float)
            z = (feats - MU) / SIGMA
            yscore = float(MODEL.predict_proba(z.reshape(1, -1))[:, 1][0])  # P(ai)

            pred_val = int(yscore >= thr_val) if use_thr else None
            meta = {"path": pth, "open_lm": args.model, "max_ctx": int(args.max_ctx)}

            collector.add(
                idx=i,
                y_true=int(y),
                y_score=yscore,
                text_len=len(txt),
                meta=meta,
                pred=pred_val,
                threshold=thr_val
            )

            # ETA
            dt = time.perf_counter() - t0
            ema_row_sec = dt if ema_row_sec is None else (alpha*dt + (1-alpha)*ema_row_sec)
            left = len(texts) - (i+1); eta = left * (ema_row_sec or 0.0)
            pbar.set_postfix({
                "r/s": f"{(1.0/(ema_row_sec or 1e-9)):.2f}",
                "avg_s/row": f"{(ema_row_sec or 0):.3f}",
                "ETA": _fmt_eta(eta),
                "fin": (datetime.now() + timedelta(seconds=eta)).strftime("%H:%M:%S") if eta>0 else "--:--:--"
            })
            pbar.update(1)
        pbar.close()

        # Write CSV
        info = collector.finalize(out_csv=str(outs[domain]), known_flip=False, threshold=thr_val)
        print(f"[DONE] {domain}: {info['out']}  collected={info['n_total']}  written={info['n_written']}")

    print("\nAll three CSVs written to:", out_dir)

if __name__ == "__main__":
    main()
