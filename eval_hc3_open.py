# eval_hc3_open.py
# Purpose: Evaluate your already-trained Ghostbuster model (trained on Ghostbuster domains)
#          OUT-OF-DOMAIN on HC3, using a local HuggingFace LM (e.g., gpt2-medium)
#          to supply token logprobs instead of any paid API.
#
# Output:  outputs/ghostbuster_open/hc3_from_gb.csv
#          with columns: id,domain,y_true,y_score,detector_name,text_len
#
# Usage:
#   (optional) set GB_OPEN_LM=gpt2-medium
#   python eval_hc3_open.py
#
# Notes:
# - Expects you already have saved model artifacts from train.py:
#       model/model        (CalibratedClassifierCV)
#       model/mu           (feature means)
#       model/sigma        (feature stds)
#       model/features.txt (selected symbolic features)
#   These are produced by:  python train.py --train_on_all_data
#   after your in-domain training.
#
# - We ALIAS both "davinci-logprobs" and "ada-logprobs" to the SAME open-LM stream
#   so your existing symbolic expressions work unchanged.

import os, csv
from pathlib import Path

import numpy as np
import dill as pickle
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from utils.featurize import normalize, t_featurize_logprobs, score_ngram
from utils.symbolic import train_trigram, get_words, vec_functions, scalar_functions

# ---- Config (override via env if you want) ----
OPEN_LM = os.getenv("GB_OPEN_LM", "gpt2-medium")   # local HF model name
MAX_CTX = int(os.getenv("GB_OPEN_MAX_CTX", "1024"))

# ---- HC3 loader ----
def load_hc3_all():
    last_err = None
    for ext in ("jsonl","json"):
        try:
            fp = hf_hub_download("Hello-SimpleAI/HC3", filename=f"all.{ext}", repo_type="dataset")
            return load_dataset("json", data_files=fp, split="train")
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not load HC3 all.jsonl/.json. Last error: {repr(last_err)}")

# ---- Open-LM token probs for a single text ----
@torch.inference_mode()
def token_probs_and_subwords(text, tok, mdl, max_ctx=1024):
    # ensure pad token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    enc = tok(
        text, return_tensors="pt", truncation=True, max_length=max_ctx
    )
    input_ids = enc["input_ids"].to(mdl.device)
    attn      = enc["attention_mask"].to(mdl.device)

    # need at least 2 tokens for shifting
    if input_ids.shape[1] < 2:
        eos_id = tok.eos_token_id if tok.eos_token_id is not None else (tok.pad_token_id or 0)
        input_ids = torch.cat([input_ids, torch.tensor([[eos_id]], device=mdl.device)], dim=1)
        attn      = torch.cat([attn, torch.ones_like(attn)], dim=1)

    out = mdl(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logits = out.logits  # [1, T, V]
    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)  # [1, T-1, V]
    next_ids = input_ids[:, 1:].unsqueeze(-1)                # [1, T-1, 1]
    tok_logp = torch.gather(logprobs, dim=-1, index=next_ids).squeeze(0).squeeze(-1)  # [T-1]
    probs = torch.exp(tok_logp).cpu().numpy()                # per-token probabilities in (0,1]

    subwords = tok.convert_ids_to_tokens(input_ids[0, 1:].tolist())
    return probs, subwords

# ---- Build one feature vector in your existing DSL ----
def build_feature_vector(text, best_features, trigram_model, enc_for_trigram, tok, mdl):
    # open-LM token probs + GPT-2 BPE subwords
    probs, subwords = token_probs_and_subwords(text, tok, mdl, MAX_CTX)

    # n-gram features (trigram, unigram) using your existing helpers
    trigram_arr = np.array(score_ngram(text, trigram_model,      enc_for_trigram, n=3, strip_first=False))
    unigram_arr = np.array(score_ngram(text, trigram_model.base, enc_for_trigram, n=1, strip_first=False))

    # t_ features want (davinci, ada, subwords): ALIAS both to our open-LM probs
    t_features = t_featurize_logprobs(probs, probs, subwords)

    # vector DSL — map names to arrays
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
                # unknown token -> skip this feature
                break

    return np.array(t_features + exp_features, dtype=float)

def main():
    # 1) Load your IN-DOMAIN trained model & normalization
    model = pickle.load(open("model/model", "rb"))
    mu    = pickle.load(open("model/mu", "rb"))
    sigma = pickle.load(open("model/sigma", "rb"))
    best_features = open("model/features.txt", "r", encoding="utf-8").read().strip().split("\n")

    # 2) Prepare the local open-LM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] Using device: {device}  |  OPEN_LM={OPEN_LM}  |  MAX_CTX={MAX_CTX}")
    tok = AutoTokenizer.from_pretrained(OPEN_LM, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(OPEN_LM).to(device).eval()
    try:
        if device == "cuda":
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # 3) Trigram model & encoder (same as your project)
    import tiktoken
    trigram_model = pickle.load(open("model/trigram_model.pkl", "rb"))
    enc_for_trigram = tiktoken.encoding_for_model("davinci").encode

    # 4) Load HC3
    ds = load_hc3_all()
    humans = [h for ex in ds for h in (ex.get("human_answers") or []) if h and h.strip()]
    ais    = [a for ex in ds for a in (ex.get("chatgpt_answers") or []) if a and a.strip()]

    # 5) Score
    rows = []
    # humans → label 0
    for i, txt in enumerate(humans):
        txt = txt.strip()
        feats = build_feature_vector(txt, best_features, trigram_model, enc_for_trigram, tok, mdl)
        feat_norm = (feats - mu) / sigma
        p = model.predict_proba(feat_norm.reshape(1, -1))[:, 1][0]
        rows.append((f"hc3_h_{i:08d}", 0, p, len(txt)))

    # ai → label 1
    for i, txt in enumerate(ais):
        txt = txt.strip()
        feats = build_feature_vector(txt, best_features, trigram_model, enc_for_trigram, tok, mdl)
        feat_norm = (feats - mu) / sigma
        p = model.predict_proba(feat_norm.reshape(1, -1))[:, 1][0]
        rows.append((f"hc3_a_{i:08d}", 1, p, len(txt)))

    # 6) Write unified CSV
    out = Path("outputs/ghostbuster_open/hc3_from_gb.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","domain","y_true","y_score","detector_name","text_len"])
        for rid, y, s, L in rows:
            w.writerow([rid, "hc3", y, f"{s:.6f}", "ghostbuster-open", L])

    print(f"[DONE] wrote {out}  |  rows={len(rows)}")

if __name__ == "__main__":
    main()
