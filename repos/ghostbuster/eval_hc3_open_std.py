# eval_hc3_open_std.py  — Ghostbuster-open on HC3 with unified CSV
# Schema: id,dataset,domain,detector,y_true,y_score,pred,threshold,text_len,meta
#
# Examples:
#   python eval_hc3_open_std.py --out outputs/unified/gbopen_hc3.csv
#   python eval_hc3_open_std.py --model gpt2 --max_ctx 512 --out outputs/unified/gbopen_hc3_gpt2.csv
#   python eval_hc3_open_std.py --no_enforce_3k --limit 2000 --out outputs/unified/gbopen_hc3_small.csv
#
# Notes:
# - y_score is the Ghostbuster-open model’s P(ai).
# - pred/threshold are blank unless --threshold >= 0 is provided.

import os, csv, time, json, argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import dill as pickle
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from utils.featurize import t_featurize_logprobs, score_ngram
from utils.symbolic import train_trigram, get_words, vec_functions, scalar_functions

# ============ unified CSV collector ============
# Minimal inlined collector to avoid extra imports; emits the exact schema.
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
        self.rows.append((idx, y_true, float(y_score), int(text_len), meta or {}, pred, threshold))

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
        # known_flip not used here; y_score already "higher => AI"
        rows = self.rows
        n_total = len(rows)
        if self.enforce:
            sel = self._balanced_indexes(3000)
            rows = [rows[i] for i in sel]
        # write
        out = Path(out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id","dataset","domain","detector","y_true","y_score","pred","threshold","text_len","meta"])
            for (idx, y_true, y_score, text_len, meta, pred_val, thr_val) in rows:
                rid = f"{self.dataset}_{self.split}_{idx}"
                pred_field = "" if pred_val is None else int(pred_val)
                thr_field = "" if thr_val is None else f"{float(thr_val):.6f}"
                w.writerow([
                    rid,
                    self.dataset,
                    self.domain,
                    self.detector,
                    int(y_true),
                    f"{float(y_score):.6f}",
                    pred_field,
                    thr_field,
                    int(text_len),
                    json.dumps(meta, ensure_ascii=False),
                ])
        return {"out": str(out), "n_total": n_total, "n_written": len(rows), "flipped": False}

# ============ args ============
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("GB_OPEN_LM", "gpt2-medium"),
                    help="HF causal LM (e.g., gpt2, gpt2-medium). Smaller = faster on CPU.")
    ap.add_argument("--max_ctx", type=int, default=int(os.getenv("GB_OPEN_MAX_CTX", "768")),
                    help="Max tokens for open LM. Reduce on CPU (e.g., 384/256).")
    ap.add_argument("--limit", type=int, default=-1,
                    help="Cap per class BEFORE 3k presampling; -1 = all.")
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"],
                    help="Force device. Default: auto-detect.")
    ap.add_argument("--out", default="outputs/unified/gbopen_hc3.csv",
                    help="Unified CSV output path.")
    ap.add_argument("--threshold", type=float, default=-1.0,
                    help="If >=0, also fill pred and threshold.")
    ap.add_argument("--no_enforce_3k", action="store_true",
                    help="If set, do not enforce 3k/3k (write as-many-as-available).")
    ap.add_argument("--strict_fail", action="store_true",
                    help="If enforcing 3k/3k and not enough data, hard fail.")
    return ap.parse_args()

# ============ data ============
def load_hc3_all():
    last_err = None
    for ext in ("jsonl","json"):
        try:
            fp = hf_hub_download("Hello-SimpleAI/HC3", filename=f"all.{ext}", repo_type="dataset")
            return load_dataset("json", data_files=fp, split="train")
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load HC3 all.jsonl/.json. Last error: {repr(last_err)}")

# ============ GB-open feature extraction ============
@torch.inference_mode()
def token_probs_and_subwords(text, tok, mdl, max_ctx: int):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_ctx)
    input_ids = enc["input_ids"].to(mdl.device)
    attn      = enc["attention_mask"].to(mdl.device)
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
    t_features = t_featurize_logprobs(probs, probs, subwords)  # alias both streams to open-LM probs
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
                exp_features.append(scalar_functions[tk](curr))
                break
            else:
                break
    return np.array(t_features + exp_features, dtype=float)

def fmt_eta(seconds: float) -> str:
    if not seconds or seconds < 0 or np.isinf(seconds) or np.isnan(seconds):
        return "--:--:--"
    seconds = int(seconds + 0.5)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# ============ main ============
def _model_tag(name: str) -> str:
    # "gpt2-medium" -> "gpt2m"; keep letters+digits
    cleaned = name.replace("-", "")
    return cleaned

def main():
    args = parse_args()

    # Load trained GB classifier + normalization
    MODEL = pickle.load(open("model/model", "rb"))
    MU    = pickle.load(open("model/mu", "rb"))
    SIGMA = pickle.load(open("model/sigma", "rb"))
    best_features = Path("model/features.txt").read_text(encoding="utf-8").strip().split("\n")

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

    # Trigram model & encoder
    import tiktoken
    trigram_model = pickle.load(open("model/trigram_model.pkl", "rb"))
    enc_for_trigram = tiktoken.encoding_for_model("davinci").encode

    # Load HC3
    ds = load_hc3_all()
    humans = [h for ex in ds for h in (ex.get("human_answers") or []) if h and h.strip()]
    ais    = [a for ex in ds for a in (ex.get("chatgpt_answers") or []) if a and a.strip()]

    # Optional per-class cap BEFORE presampling (debug speed-up)
    if args.limit and args.limit > 0:
        humans = humans[:args.limit]
        ais    = ais[:args.limit]

    # Enforce/relax 3k/3k **before** scoring (fast)
    need = 3000
    enforce = not args.no_enforce_3k
    if enforce:
        if len(humans) < need or len(ais) < need:
            msg = f"[WARN] HC3 insufficient for 3k/3k: human={len(humans)}, ai={len(ais)}."
            if args.strict_fail:
                raise RuntimeError(msg)
            print(msg, "Proceeding with as-many-as-available.")
            enforce = False
        else:
            rng = np.random.default_rng(42)
            humans = humans[:need] if len(humans) == need else humans.copy()
            ais    = ais[:need]    if len(ais)    == need else ais.copy()
            if len(humans) > need:
                humans = list(rng.choice(humans, size=need, replace=False))
            if len(ais) > need:
                ais = list(rng.choice(ais, size=need, replace=False))
            print("[info] Presampled to exactly 3k/3k prior to inference.")

    total = len(humans) + len(ais)
    print(f"[info] Humans: {len(humans)} | AI: {len(ais)} | Total to score: {total}")

    # Unified collector
    detector = f"ghostbuster-open-{_model_tag(args.model)}"
    collector = StdCollector(
        dataset="hc3",
        domain="hc3-all",
        detector=detector,
        split="all",
        random_state=42,
        enforce_3k_per_class=False  # we already presampled if enforcing
    )

    # Thresholding?
    use_threshold = (args.threshold is not None) and (args.threshold >= 0.0)
    thr_val = args.threshold if use_threshold else None

    # Process helper
    def score_side(texts, y_true, prefix):
        ema_row_sec = None
        alpha = 0.2
        pbar = tqdm(total=len(texts), desc=f"{prefix} ({'human' if y_true==0 else 'ai'})", unit="doc")
        for i, txt in enumerate(texts):
            t0 = time.perf_counter()
            txt = (txt or "").strip()
            feats = build_feature_vector(txt, best_features, trigram_model, enc_for_trigram, tok, mdl, args.max_ctx)
            y_score = float(MODEL.predict_proba(((feats - MU) / SIGMA).reshape(1, -1))[:, 1][0])  # P(ai)

            pred_field = int(y_score >= thr_val) if use_threshold else None
            meta = {
                "open_lm": args.model,
                "max_ctx": int(args.max_ctx),
                "feature_file": "model/features.txt",
                "trigram_file": "model/trigram_model.pkl",
                "gb_model_files": ["model/model","model/mu","model/sigma"],
                "side_prefix": prefix
            }
            # idx: keep stable within side by i
            collector.add(
                idx=i if y_true==0 else (10_000_000 + i),  # disambiguate human/ai spaces
                y_true=y_true,
                y_score=y_score,
                text_len=len(txt),
                meta=meta,
                pred=pred_field,
                threshold=thr_val
            )

            # Progress ETA
            dt = time.perf_counter() - t0
            ema_row_sec = dt if ema_row_sec is None else (alpha*dt + (1-alpha)*ema_row_sec)
            left = len(texts) - (i+1)
            eta = left * (ema_row_sec or 0.0)
            rps = 1.0 / ema_row_sec if ema_row_sec and ema_row_sec > 0 else 0.0
            pbar.set_postfix({"r/s": f"{rps:.2f}", "avg_s/row": f"{(ema_row_sec or 0):.3f}",
                              "ETA": fmt_eta(eta),
                              "fin": (datetime.now() + timedelta(seconds=eta)).strftime("%H:%M:%S") if eta>0 else "--:--:--"})
            pbar.update(1)
        pbar.close()

    # Run both sides
    score_side(humans, 0, "hc3_h")
    score_side(ais,    1, "hc3_a")

    # Finalize unified CSV
    info = collector.finalize(out_csv=args.out, known_flip=False, threshold=thr_val)
    print(f"[DONE] Unified CSV: {info['out']}  n_total_collected={info['n_total']}  n_written={info['n_written']}")
    if enforce:
        print("Enforced 3k/3k via presampling.")
    else:
        print("Wrote as-many-as-available (no 3k/3k enforcement).")

if __name__ == "__main__":
    main()
