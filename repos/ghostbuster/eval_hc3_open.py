# eval_hc3_open.py  — CPU/GPU friendly with progress + dynamic ETA
import os, csv, time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import dill as pickle
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import argparse

from utils.featurize import t_featurize_logprobs, score_ngram
from utils.symbolic import train_trigram, get_words, vec_functions, scalar_functions

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("GB_OPEN_LM", "gpt2-medium"),
                    help="HF causal LM (e.g., gpt2, gpt2-medium). Smaller = faster on CPU.")
    ap.add_argument("--max_ctx", type=int, default=int(os.getenv("GB_OPEN_MAX_CTX", "768")),
                    help="Max tokens for open LM. Reduce on CPU (e.g., 384/256).")
    ap.add_argument("--limit", type=int, default=-1,
                    help="Limit #docs per split (humans/ai) for quick tests; -1 = all.")
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda"],
                    help="Force device. Default: auto-detect.")
    ap.add_argument("--out", default="outputs/ghostbuster_open/hc3_from_gb.csv",
                    help="Output CSV path.")
    return ap.parse_args()

def load_hc3_all():
    last_err = None
    for ext in ("jsonl","json"):
        try:
            fp = hf_hub_download("Hello-SimpleAI/HC3", filename=f"all.{ext}", repo_type="dataset")
            return load_dataset("json", data_files=fp, split="train")
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load HC3 all.jsonl/.json. Last error: {repr(last_err)}")

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
    t_features = t_featurize_logprobs(probs, probs, subwords)  # alias both gpt streams to open-LM probs
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

def score_split(texts, label, prefix, best_features, trigram_model, enc_for_trigram, tok, mdl, max_ctx, writer):
    # EMA-based ETA
    ema_row_sec = None
    alpha = 0.2
    start_time = time.perf_counter()
    pbar = tqdm(total=len(texts), desc=f"{prefix} ({'human' if label==0 else 'ai'})", unit="doc")
    written = 0

    for i, txt in enumerate(texts):
        t0 = time.perf_counter()
        rid = f"{prefix}_{i:08d}"
        txt = (txt or "").strip()
        feats = build_feature_vector(txt, best_features, trigram_model, enc_for_trigram, tok, mdl, max_ctx)
        pred = MODEL.predict_proba(((feats - MU) / SIGMA).reshape(1, -1))[:, 1][0]
        writer.writerow([rid, "hc3", label, f"{pred:.6f}", "ghostbuster-open", len(txt)])
        written += 1

        dt = time.perf_counter() - t0
        if ema_row_sec is None:
            ema_row_sec = dt
        else:
            ema_row_sec = alpha*dt + (1-alpha)*ema_row_sec

        left = len(texts) - (i+1)
        eta = left * (ema_row_sec or 0.0)
        finish = (datetime.now() + timedelta(seconds=eta)).strftime("%H:%M:%S")
        rps = 1.0 / ema_row_sec if ema_row_sec and ema_row_sec > 0 else 0.0
        pbar.set_postfix({
            "r/s": f"{rps:.2f}",
            "avg_s/row": f"{(ema_row_sec or 0):.3f}",
            "ETA": fmt_eta(eta),
            "fin": finish
        })
        pbar.update(1)

    pbar.close()
    total_dt = time.perf_counter() - start_time
    print(f"[split done] {prefix} wrote {written} rows in {fmt_eta(total_dt)} "
          f"(~{(written/total_dt if total_dt>0 else 0):.2f} r/s)")
    return written

def main():
    global MODEL, MU, SIGMA  # used inside score_split
    args = parse_args()

    # Load trained model & normalization (from your in-domain training)
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

    if args.limit and args.limit > 0:
        humans = humans[:args.limit]
        ais    = ais[:args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","domain","y_true","y_score","detector_name","text_len"])

        print(f"[info] Humans: {len(humans)} | AI: {len(ais)} | Output: {out}")
        wrote_h = score_split(humans, 0, "hc3_h", best_features, trigram_model, enc_for_trigram, tok, mdl, args.max_ctx, w)
        wrote_a = score_split(ais,    1, "hc3_a", best_features, trigram_model, enc_for_trigram, tok, mdl, args.max_ctx, w)

    print(f"[DONE] wrote {wrote_h + wrote_a} rows → {out}")

if __name__ == "__main__":
    main()
