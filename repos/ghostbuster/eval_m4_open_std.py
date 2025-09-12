#!/usr/bin/env python
# -*- coding: utf-8 -*-

# eval_m4_open_std.py — Ghostbuster-open on M4 (paired human/machine)
# Unified CSV schema:
# id,dataset,domain,detector,y_true,y_score,pred,threshold,text_len,meta

import os, csv, json, glob, time, argparse, random, warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
import dill as pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# GB feature bits
from utils.featurize import t_featurize_logprobs, score_ngram
from utils.symbolic import train_trigram, get_words, vec_functions, scalar_functions

# -------------------- Unified CSV collector --------------------
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
        sel = np.concatenate([sel_h, sel_a])
        self.rng.shuffle(sel)
        return sel

    def finalize(self, out_csv, known_flip=False, threshold=None):
        rows = self.rows
        if self.enforce:
            sel = self._balanced_indexes(3000)
            rows = [rows[i] for i in sel]

        out = Path(out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id","dataset","domain","detector","y_true","y_score","pred","threshold","text_len","meta"])
            for (idx, y_true, y_score, text_len, meta, pred_val, thr_val) in rows:
                rid = f"{self.dataset}_{self.split}_{idx}"
                pred_field = "" if pred_val is None else int(pred_val)
                thr_field  = "" if thr_val  is None else f"{float(thr_val):.6f}"
                w.writerow([
                    rid, self.dataset, self.domain, self.detector,
                    int(y_true), f"{float(y_score):.6f}", pred_field, thr_field,
                    int(text_len), json.dumps(meta, ensure_ascii=False)
                ])
        return {"out": str(out), "n_total": len(self.rows), "n_written": len(rows), "flipped": False}

# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="Folder containing M4 *.jsonl files (paired human_text/machine_text).")
    ap.add_argument("--include", nargs="*", default=None,
                    help="Optional filename substrings to keep (e.g. arxiv_ chatgpt davinci cohere flant5 dolly).")
    ap.add_argument("--english_only", action="store_true",
                    help="Heuristic English filter (ASCII-heavy).")
    ap.add_argument("--min_chars", type=int, default=80,
                    help="Drop texts shorter than this many characters.")
    ap.add_argument("--limit_per_class", type=int, default=-1,
                    help="Cap examples per class after all filters; -1 = no cap.")

    ap.add_argument("--model", default=os.getenv("GB_OPEN_LM", "gpt2-medium"),
                    help="HF causal LM for surrogate logprobs (e.g., gpt2, gpt2-medium).")
    ap.add_argument("--max_ctx", type=int, default=int(os.getenv("GB_OPEN_MAX_CTX", "768")),
                    help="Max tokens fed into open LM (reduce on CPU: 384/256).")
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"],
                    help="Force device; default auto.")
    ap.add_argument("--out", default="outputs/unified/gbopen_m4.csv",
                    help="Unified CSV output path.")
    ap.add_argument("--threshold", type=float, default=-1.0,
                    help="Optional decision threshold (>=0 enables pred/threshold).")
    ap.add_argument("--no_enforce_3k", action="store_true",
                    help="If set, do not enforce 3k/3k (write as-many-as-available).")
    ap.add_argument("--strict_fail", action="store_true",
                    help="If enforcing 3k/3k and not enough data, hard fail.")
    ap.add_argument("--domain_tag", default="m4-all",
                    help='Domain tag in CSV, default "m4-all" (e.g., use "m4-arxiv" if filtering).')
    return ap.parse_args()

# -------------------- Helpers --------------------
def is_english_text(s: str) -> bool:
    if not isinstance(s, str): return False
    sample = s[:1000]
    non_ascii = sum(1 for ch in sample if ord(ch) > 127)
    return non_ascii <= len(sample) * 0.05

def discover_jsonl_files(root: str, include: Optional[List[str]]) -> List[str]:
    files = sorted(glob.glob(os.path.join(root, "*.jsonl")))
    if include:
        inc = [p.lower() for p in include]
        files = [f for f in files if any(p in os.path.basename(f).lower() for p in inc)]
    return files

# ---- robust coercion (handles str/list/dict + alt keys) ----
def _coerce_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return " ".join(_coerce_text(y) for y in x if y is not None)
    if isinstance(x, dict):
        # Prefer known text-bearing keys
        for k in ("text", "content", "answer", "completion", "response", "abstract", "body"):
            if k in x:
                return _coerce_text(x[k])
        try:
            return " ".join(_coerce_text(v) for v in x.values())
        except Exception:
            return ""
    return str(x)

def _first_nonempty(obj: dict, keys: List[str]) -> str:
    for k in keys:
        if k in obj:
            s = _coerce_text(obj[k]).strip()
            if s:
                return s
    return ""

def load_m4_paired(files: List[str], english_only: bool, min_chars: int):
    """
    Emit records with fields:
      {"text": <str>, "y": 0/1, "file": <path>}
    Accepts varied shapes and alt key names across M4 files.
    """
    recs = []
    human_keys   = ["human_text", "abstract", "reference", "source_text", "human", "gold"]
    machine_keys = ["machine_text", "machine_abstract", "generated_text", "model_output", "prediction", "answer"]

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    continue

                h = _first_nonempty(obj, human_keys)
                m = _first_nonempty(obj, machine_keys)

                # generic fallbacks
                if not h:
                    h = _coerce_text(obj.get("prompt", "")).strip()
                if not m:
                    m = _coerce_text(obj.get("completion", obj.get("response", ""))).strip()

                # filters
                if min_chars:
                    if len(h) < min_chars: h = ""
                    if len(m) < min_chars: m = ""
                if english_only:
                    if h and not is_english_text(h): h = ""
                    if m and not is_english_text(m): m = ""

                if h:
                    recs.append({"text": h, "y": 0, "file": fp})
                if m:
                    recs.append({"text": m, "y": 1, "file": fp})

    return recs

@torch.inference_mode()
def token_probs_and_subwords(text, tok, mdl, max_ctx: int):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_ctx)
    input_ids = enc["input_ids"].to(mdl.device)
    attn      = enc["attention_mask"].to(mdl.device)

    # Ensure at least 2 tokens to align next-token probs
    if input_ids.shape[1] < 2:
        eos_id = tok.eos_token_id if tok.eos_token_id is not None else (tok.pad_token_id or 0)
        input_ids = torch.cat([input_ids, torch.tensor([[eos_id]], device=mdl.device)], dim=1)
        attn      = torch.cat([attn, torch.ones_like(attn)], dim=1)

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
    t_features = t_featurize_logprobs(probs, probs, subwords)  # surrogate for ada/davinci streams

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
                nxt = vector_map[tokens[i+1]]
                curr = vec_functions[tk](curr, nxt)
                i += 2
            elif tk in scalar_functions:
                exp_features.append(float(scalar_functions[tk](curr)))
                break
            else:
                break

    return np.array(t_features + exp_features, dtype=float)

def fmt_eta(seconds: float) -> str:
    if not seconds or seconds < 0 or np.isinf(seconds) or np.isnan(seconds):
        return "--:--:--"
    seconds = int(seconds + 0.5); h = seconds // 3600; m = (seconds % 3600) // 60; s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# -------------------- Main --------------------
def _detector_tag(model_name: str) -> str:
    # "gpt2-medium" -> "gpt2medium"
    return model_name.replace("-", "")

def main():
    global MODEL, MU, SIGMA
    args = parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("default")

    # Load GB classifier + normalization + feature list
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
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    try:
        if device == "cuda":
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Load M4 pairs
    files = discover_jsonl_files(args.data_root, args.include)
    if not files:
        raise FileNotFoundError("[ERROR] No .jsonl files found with current --data_root/--include settings.")
    print(f"[info] Files considered: {len(files)}")
    recs = load_m4_paired(files, english_only=args.english_only, min_chars=args.min_chars)
    print(f"[info] Records after filters: {len(recs)}")
    human = [r for r in recs if r["y"] == 0]
    ai    = [r for r in recs if r["y"] == 1]
    random.shuffle(human); random.shuffle(ai)

    if args.limit_per_class and args.limit_per_class > 0:
        human = human[:args.limit_per_class]
        ai    = ai[:args.limit_per_class]

    n_h, n_a = len(human), len(ai)
    print(f"[info] Class counts → human={n_h} | ai={n_a}")

    # --- presample to exactly 3k/3k BEFORE inference (fast) ---
    need = 3000
    enforce = not args.no_enforce_3k
    if enforce:
        if n_h < need or n_a < need:
            msg = f"[WARN] Not enough for 3k/3k after filters (human={n_h}, ai={n_a})."
            if args.strict_fail:
                raise RuntimeError(msg)
            print(msg, "Proceeding with as-many-as-available.")
        else:
            rng = np.random.default_rng(42)
            human = list(rng.choice(human, size=need, replace=False))
            ai    = list(rng.choice(ai,    size=need, replace=False))
            n_h, n_a = len(human), len(ai)
            print("[info] Presampled to exactly 3k/3k prior to inference.")

    rows = human + ai
    random.shuffle(rows)
    print(f"[info] Will score {len(rows)} rows total.")

    # Unified collector (dataset/domain/detector)
    detector = f"ghostbuster-open-{_detector_tag(args.model)}"
    collector = StdCollector(
        dataset="m4",
        domain=args.domain_tag,
        detector=detector,
        split="all",
        random_state=42,
        enforce_3k_per_class=False  # we presampled already if enforcing
    )

    # threshold usage?
    use_thr = (args.threshold is not None) and (args.threshold >= 0.0)
    thr_val = args.threshold if use_thr else None

    # --- scoring loop with ETA ---
    ema_row_sec, alpha = None, 0.2
    pbar = tqdm(total=len(rows), desc=f"GB-open on M4", unit="doc")
    for i, ex in enumerate(rows):
        t0 = time.perf_counter()
        txt = (ex["text"] or "").strip()
        y   = int(ex["y"])
        file_path = ex["file"]

        feats = build_feature_vector(txt, best_features, trigram_model, enc_for_trigram, tok, mdl, args.max_ctx)
        z = (feats - MU) / SIGMA
        yscore = float(MODEL.predict_proba(z.reshape(1, -1))[:, 1][0])  # P(ai)

        pred_val = int(yscore >= thr_val) if use_thr else None
        meta = {"path": file_path, "open_lm": args.model, "max_ctx": int(args.max_ctx)}

        # idx: use running index; ids are stable within this run
        collector.add(
            idx=i,
            y_true=y,
            y_score=yscore,
            text_len=len(txt),
            meta=meta,
            pred=pred_val,
            threshold=thr_val
        )

        dt = time.perf_counter() - t0
        ema_row_sec = dt if ema_row_sec is None else (alpha*dt + (1-alpha)*ema_row_sec)
        left = len(rows) - (i+1)
        eta  = left * (ema_row_sec or 0.0)
        pbar.set_postfix({
            "r/s": f"{(1.0/(ema_row_sec or 1e-9)):.2f}",
            "avg_s/row": f"{(ema_row_sec or 0):.3f}",
            "ETA": f"{int(eta//3600):02d}:{int((eta%3600)//60):02d}:{int(eta%60):02d}",
            "fin": (datetime.now() + timedelta(seconds=eta)).strftime("%H:%M:%S") if eta>0 else "--:--:--",
        })
        pbar.update(1)
    pbar.close()

    info = collector.finalize(out_csv=args.out, known_flip=False, threshold=thr_val)
    print(f"[DONE] Unified CSV: {info['out']} | collected={info['n_total']} | written={info['n_written']}")
    if enforce and info['n_written'] == 6000:
        print("Enforced 3k/3k via presampling.")
    elif enforce:
        print("Presampling requested but insufficient data; wrote as-many-as-available.")

if __name__ == "__main__":
    main()
