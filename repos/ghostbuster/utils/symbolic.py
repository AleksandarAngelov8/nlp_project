# utils/symbolic.py
from nltk.util import ngrams
from nltk.corpus import brown
from nltk.tokenize import word_tokenize  # retained for compatibility

import os, json, hashlib
import tqdm
import numpy as np
import tiktoken
import dill as pickle

from utils.featurize import *
from utils.n_gram import *

from collections import defaultdict

# =========================
# Robust vector operations
# =========================

def _to_np(x):
    x = np.asarray(x)
    if x.ndim != 1:
        x = x.reshape(-1,)
    return x

def _pad_to(a, n):
    a = _to_np(a)
    if a.shape[0] == n:
        return a
    if a.shape[0] < n:
        return np.pad(a, (0, n - a.shape[0]), mode="constant", constant_values=0.0)
    return a[:n]

def _align(a, b, mode="max"):
    """
    Align two vectors for element-wise ops.
    mode="max": pad both to the longer length (default).
    mode="min": truncate both to the shorter length.
    """
    a = _to_np(a); b = _to_np(b)
    n = max(a.shape[0], b.shape[0]) if mode == "max" else min(a.shape[0], b.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    return _pad_to(a, n), _pad_to(b, n)

def _safe_div(a, b, eps=1e-8):
    A, B = _align(a, b, mode="max")
    B = np.where(np.abs(B) < eps, np.sign(B) * eps + (B == 0) * eps, B)
    return A / B

vec_functions = {
    "v-add": lambda a, b: (lambda A, B: A + B)(*_align(a, b, mode="max")),
    "v-sub": lambda a, b: (lambda A, B: A - B)(*_align(a, b, mode="max")),
    "v-mul": lambda a, b: (lambda A, B: A * B)(*_align(a, b, mode="max")),
    "v-div": lambda a, b: _safe_div(a, b),
    "v->":   lambda a, b: (lambda A, B: A > B)(*_align(a, b, mode="max")),
    "v-<":   lambda a, b: (lambda A, B: A < B)(*_align(a, b, mode="max")),
}

# =========================
# Safe scalar reducers
# =========================

def _safe_max(x):
    x = list(x)
    return max(x) if len(x) else 0.0

def _safe_min(x):
    x = list(x)
    return min(x) if len(x) else 0.0

def _safe_avg(x):
    x = list(x)
    return (sum(x) / len(x)) if len(x) else 0.0

def _safe_avg_top_k(x, k=25):
    x = sorted(list(x), reverse=True)
    if not x:
        return 0.0
    k = min(k, len(x))
    return sum(x[:k]) / k

def _safe_var(x):
    x = np.asarray(list(x))
    return float(np.var(x)) if x.size else 0.0

def _safe_l2(x):
    x = np.asarray(list(x))
    return float(np.linalg.norm(x)) if x.size else 0.0

scalar_functions = {
    "s-max": _safe_max,
    "s-min": _safe_min,
    "s-avg": _safe_avg,
    "s-avg-top-25": lambda x: _safe_avg_top_k(x, 25),
    "s-len": lambda x: len(list(x)),
    "s-var": _safe_var,
    "s-l2":  _safe_l2,
}

# =========================
# Feature expression space
# =========================

vectors = ["davinci-logprobs", "ada-logprobs", "trigram-logprobs", "unigram-logprobs"]

vec_combinations = defaultdict(list)
for i in range(len(vectors)):
    for j in range(i):
        for func in vec_functions:
            if func != "v-div":
                vec_combinations[vectors[i]].append(f"{func} {vectors[j]}")
for v1 in vectors:
    for v2 in vectors:
        if v1 != v2:
            vec_combinations[v1].append(f"v-div {v2}")

def get_words(exp):
    return exp.split(" ")

def backtrack_functions(
    vectors=("davinci-logprobs", "ada-logprobs", "trigram-logprobs", "unigram-logprobs"),
    max_depth=2,
):
    def helper(prev, depth):
        if depth >= max_depth:
            return []
        out = []
        prev_word = get_words(prev)[-1]
        for func in scalar_functions:
            out.append(f"{prev} {func}")
        for comb in vec_combinations[prev_word]:
            out += helper(f"{prev} {comb}", depth + 1)
        return out

    ret = []
    for vec in vectors:
        ret += helper(vec, 0)
    return ret

# =========================
# Trigram model
# =========================

def train_trigram(verbose=True, return_tokenizer=False):
    enc = tiktoken.encoding_for_model("davinci")
    tokenizer = enc.encode

    sentences = brown.sents()
    if verbose:
        print("Tokenizing corpus...")
    tokenized_corpus = []
    for sentence in tqdm.tqdm(sentences):
        tokenized_corpus += tokenizer(" ".join(sentence))

    if verbose:
        print("\nTraining n-gram model...")

    if return_tokenizer:
        return TrigramBackoff(tokenized_corpus), tokenizer
    else:
        return TrigramBackoff(tokenized_corpus)

# =========================
# Logprob collection
# =========================

def _read_text_utf8(path):
    for enc in ("utf-8", "utf-8-sig"):
        try:
            with open(path, "r", encoding=enc) as f:
                s = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            s = f.read()
    return s.replace("\r\n", "\n").replace("\r", "\n")

def get_all_logprobs(
    generate_dataset,
    preprocess=lambda x: x.strip(),
    verbose=True,
    trigram=None,
    tokenizer=None,
    num_tokens=2047,
):
    if trigram is None:
        trigram, tokenizer = train_trigram(verbose=verbose, return_tokenizer=True)

    davinci_logprobs, ada_logprobs = {}, {}
    trigram_logprobs, unigram_logprobs = {}, {}

    if verbose:
        print("Loading logprobs into memory")

    file_names = generate_dataset(lambda file: file, verbose=False)
    it = tqdm.tqdm(file_names) if verbose else file_names

    for file in it:
        if "logprobs" in file.lower():
            continue

        doc = preprocess(_read_text_utf8(file))

        davinci_logprobs[file] = get_logprobs(
            convert_file_to_logprob_file(file, "davinci")
        )[:num_tokens]

        ada_logprobs[file] = get_logprobs(
            convert_file_to_logprob_file(file, "ada")
        )[:num_tokens]

        trigram_logprobs[file] = score_ngram(doc, trigram, tokenizer, n=3)[:num_tokens]
        unigram_logprobs[file] = score_ngram(doc, trigram.base, tokenizer, n=1)[:num_tokens]

    return davinci_logprobs, ada_logprobs, trigram_logprobs, unigram_logprobs

# =========================
# Symbolic feature generation
# =========================

def _exp_sha1(exp: str) -> str:
    return hashlib.sha1(exp.encode("utf-8")).hexdigest()

def _load_meta(meta_path):
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"features": [], "rows": None}

def _save_meta(meta_path, meta):
    tmp = meta_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp, meta_path)

def generate_symbolic_data(
    generate_dataset,
    preprocess=lambda x: x,
    max_depth=2,
    output_file="symbolic_data",
    verbose=True,
    vector_map=None,
    checkpoint_every=100,          # NEW: how often to flush meta
    checkpoint_dir=None,           # NEW: where to write per-feature npy
    resume=True,                   # NEW: resume from existing checkpoint
):
    """
    Brute force and cache symbolic features.
    Now with checkpointing: each feature column saved as .npy in checkpoint_dir,
    with a meta.json manifest. At the end, consolidates to a single pickle
    named `output_file` (same as before).
    """
    if vector_map is None:
        (
            davinci_logprobs,
            ada_logprobs,
            trigram_logprobs,
            unigram_logprobs,
        ) = get_all_logprobs(generate_dataset, preprocess=preprocess, verbose=verbose)

        vector_map = {
            "davinci-logprobs": lambda file: davinci_logprobs[file],
            "ada-logprobs": lambda file: ada_logprobs[file],
            "trigram-logprobs": lambda file: trigram_logprobs[file],
            "unigram-logprobs": lambda file: unigram_logprobs[file],
        }

        all_funcs = backtrack_functions(max_depth=max_depth)
    else:
        all_funcs = backtrack_functions(max_depth=max_depth)

    if verbose:
        print(f"\nTotal # of Features: {len(all_funcs)}.")
        print("Sampling 5 features:")
        for i in range(min(5, len(all_funcs))):
            print(all_funcs[np.random.randint(0, len(all_funcs))])
        print("\nGenerating datasets...")

    # determine dataset length once (for shape consistency)
    # We compute a trivial column to get the row count.
    dummy = generate_dataset(lambda file: 0)
    N = len(dummy)

    # Prepare checkpointing
    if checkpoint_dir is None:
        checkpoint_dir = f"{output_file}_ckpt"
    os.makedirs(checkpoint_dir, exist_ok=True)
    meta_path = os.path.join(checkpoint_dir, "meta.json")
    meta = _load_meta(meta_path)
    if meta.get("rows") is None:
        meta["rows"] = N
    elif meta.get("rows") != N:
        raise RuntimeError(f"Checkpoint row count {meta['rows']} != current dataset rows {N}")

    done = set(ent["hash"] for ent in meta["features"])

    def calc_features(file, exp):
        exp_tokens = get_words(exp)
        curr = vector_map[exp_tokens[0]](file)

        i = 1
        while i < len(exp_tokens):
            tok = exp_tokens[i]
            if tok in vec_functions:
                next_vec = vector_map[exp_tokens[i + 1]](file)
                curr = vec_functions[tok](curr, next_vec)
                i += 2
            elif tok in scalar_functions:
                # Safe scalar reduce (handles empty sequences)
                return scalar_functions[tok](curr)
            else:
                break
        # If no scalar applied, reduce to mean as a stable default.
        v = _to_np(curr)
        return float(v.mean()) if v.size else 0.0

    wrote_since_flush = 0
    pbar = tqdm.tqdm(all_funcs, desc="Generating features", unit="feat")
    for exp in pbar:
        h = _exp_sha1(exp)
        npy_path = os.path.join(checkpoint_dir, f"{h}.npy")
        if resume and (h in done or os.path.isfile(npy_path)):
            # ensure meta entry exists if file exists
            if h not in done and os.path.isfile(npy_path):
                meta["features"].append({"hash": h, "expr": exp, "path": npy_path})
                done.add(h)
                wrote_since_flush += 1
            continue

        # compute column for this expression
        col = np.asarray(generate_dataset(lambda file: calc_features(file, exp)), dtype=np.float32)
        if col.shape[0] != N:
            # keep consistent shape; pad or truncate
            if col.shape[0] < N:
                col = np.pad(col, (0, N - col.shape[0]), mode="constant", constant_values=0.0)
            else:
                col = col[:N]

        # save as checkpoint
        np.save(npy_path, col)
        meta["features"].append({"hash": h, "expr": exp, "path": npy_path})
        done.add(h)
        wrote_since_flush += 1

        if wrote_since_flush >= checkpoint_every:
            _save_meta(meta_path, meta)
            wrote_since_flush = 0

    # final meta flush
    _save_meta(meta_path, meta)

    # Consolidate into the original pickle format: dict expr -> (N x 1) column
    if verbose:
        print("Consolidating checkpoints into pickle:", output_file)
    exp_to_data = {}
    for ent in tqdm.tqdm(meta["features"], desc="Packing", unit="feat"):
        col = np.load(ent["path"]).reshape(-1, 1)
        exp_to_data[ent["expr"]] = col

    pickle.dump(exp_to_data, open(output_file, "wb"))

def get_exp_featurize(best_features, vector_map):
    """
    Build a featurizer that computes a vector of the selected symbolic features.
    """
    def calc_features(file, exp):
        exp_tokens = get_words(exp)
        curr = vector_map[exp_tokens[0]](file)

        i = 1
        while i < len(exp_tokens):
            tok = exp_tokens[i]
            if tok in vec_functions:
                next_vec = vector_map[exp_tokens[i + 1]](file)
                curr = vec_functions[tok](curr, next_vec)
                i += 2
            elif tok in scalar_functions:
                return scalar_functions[tok](curr)
            else:
                break
        v = _to_np(curr)
        return float(v.mean()) if v.size else 0.0

    def exp_featurize(file):
        return np.array([calc_features(file, exp) for exp in best_features])

    return exp_featurize
