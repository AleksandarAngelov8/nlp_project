# utils/featurize.py
import numpy as np
import os
import tqdm
from nltk import ngrams
from utils.score import k_fold_score

# ---------- Robust file readers (UTF-8 + safe fallback) ----------

def _read_text_lines(path):
    """
    Read a text file as UTF-8, ignoring undecodable bytes.
    Returns a list of non-empty lines (stripped).
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    # Normalize newlines and strip
    lines = [ln for ln in txt.replace("\r\n", "\n").replace("\r", "\n").split("\n") if ln]
    return lines

# ---------- Low-level helpers ----------

def get_logprobs(file):
    """
    Returns a vector containing token probabilities (exp of negative logprob)
    from a given logprobs file.
    Each line is expected to be: "<token> <neg_logprob>"
    """
    logprobs = []
    lines = _read_text_lines(file)
    for line in lines:
        parts = line.split(" ")
        if len(parts) < 2:
            # Malformed line; skip
            continue
        try:
            # Stored as negative logprob -> convert to prob in (0,1]
            p = np.exp(-float(parts[1]))
            # Clamp to a safe numeric range just in case
            if np.isfinite(p) and p > 0:
                logprobs.append(p)
        except Exception:
            # Skip lines that can't be parsed
            continue
    return np.array(logprobs, dtype=float)

def get_tokens(file):
    """
    Returns a list of all tokens from a given logprobs file.
    Each line is: "<token> <neg_logprob>"
    """
    tokens = []
    lines = _read_text_lines(file)
    for line in lines:
        parts = line.split(" ")
        if not parts:
            continue
        tokens.append(parts[0])
    return tokens

def get_token_len(tokens):
    """
    Returns a vector of word lengths, in tokens, using GPT-2 BPE convention:
    A leading 'Ġ' marks a new wordpiece sequence.
    """
    tokens_len = []
    curr = 0
    for token in tokens:
        if token and token[0] == "Ġ":
            # New word starts; close the previous length (if any)
            tokens_len.append(curr)
            curr = 1
        else:
            curr += 1
    # If the text didn't end on a boundary, include the last segment
    if curr > 0:
        tokens_len.append(curr)
    return np.array(tokens_len, dtype=float) if tokens_len else np.array([], dtype=float)

def get_diff(file1, file2):
    """
    Returns difference in per-token probabilities between file1 and file2:
    prob(file1) - prob(file2)
    """
    return get_logprobs(file1) - get_logprobs(file2)

def convolve(X, window=100):
    """
    Returns a running average (window size). Unused in training but kept for parity.
    """
    if window <= 0 or len(X) == 0:
        return np.array([], dtype=float)
    if len(X) < window:
        # Not enough length: return a single mean
        return np.array([float(np.mean(X))], dtype=float)
    ret = []
    for i in range(len(X) - window):
        ret.append(np.mean(X[i : i + window]))
    return np.array(ret, dtype=float)

def score_ngram(doc, model, tokenizer, n=3, strip_first=False):
    """
    Returns vector of n-gram probabilities for the document using a provided model/tokenizer.
    tokenizer(doc) should return integer token ids (GPT-2 BPE ids).
    """
    scores = []
    if strip_first:
        doc = " ".join(doc.split()[:1000])
    # GPT-2 end-of-text id = 50256; pad n-1 tokens at start
    prefix = (n - 1) * [50256]
    for tup in ngrams(prefix + tokenizer(doc.strip()), n):
        try:
            scores.append(model.n_gram_probability(tup))
        except Exception:
            # If model throws for an n-gram, skip it
            continue
    return np.array(scores, dtype=float)

# ---------- Normalization & feature plumbing ----------

def normalize(data, mu=None, sigma=None, ret_mu_sigma=False):
    """
    Z-normalize a 2D array (N x D). If sigma component is zero, leave it at 1 to avoid div-by-zero.
    """
    data = np.asarray(data, dtype=float)
    if mu is None:
        mu = np.mean(data, axis=0)
    if sigma is None:
        raw_std = np.std(data, axis=0)
        sigma = np.ones_like(raw_std)
        nz = raw_std != 0
        sigma[nz] = raw_std[nz]
    out = (data - mu) / sigma
    if ret_mu_sigma:
        return out, mu, sigma
    return out

def convert_file_to_logprob_file(file_name, model):
    """
    Removes the extension of file_name, then goes to the logprobs folder of the current directory,
    and appends a -{model}.txt to it.
    Example: convert_file_to_logprob_file("data/test.txt", "davinci") = "data/logprobs/test-davinci.txt"
    """
    directory = os.path.dirname(file_name)
    base_name = os.path.basename(file_name)
    file_name_without_ext = os.path.splitext(base_name)[0]
    logprob_directory = os.path.join(directory, "logprobs")
    logprob_file_name = f"{file_name_without_ext}-{model}.txt"
    logprob_file_path = os.path.join(logprob_directory, logprob_file_name)
    return logprob_file_path

# ---------- Handcrafted T-features ----------

def t_featurize_logprobs(davinci_logprobs, ada_logprobs, tokens):
    """
    Build the classic 't_' feature vector from per-token probabilities and token strings.
    """
    X = []

    # outliers > 3 (these are probabilities, so this condition rarely holds; retain original logic)
    outliers = [lp for lp in davinci_logprobs if lp > 3]
    X.append(len(outliers))
    outliers = (outliers + [0] * 50)[:50]
    X.append(np.mean(outliers[:25]) if outliers[:25] else 0.0)
    X.append(np.mean(outliers[25:50]) if outliers[25:50] else 0.0)

    diffs = sorted((davinci_logprobs - ada_logprobs).tolist(), reverse=True)
    diffs = (diffs + [0] * 50)[:50]
    X.append(np.mean(diffs[:25]) if diffs[:25] else 0.0)
    X.append(np.mean(diffs[25:50]) if diffs[25:50] else 0.0)

    token_len = sorted(get_token_len(tokens).tolist(), reverse=True)
    token_len = (token_len + [0] * 50)[:50]
    X.append(np.mean(token_len[:25]) if token_len[:25] else 0.0)
    X.append(np.mean(token_len[25:50]) if token_len[25:50] else 0.0)

    return X

def t_featurize(file, num_tokens=2048):
    """
    Manually handcrafted features for classification, computed from saved logprob files.
    """
    davinci_file = convert_file_to_logprob_file(file, "davinci")
    ada_file     = convert_file_to_logprob_file(file, "ada")

    # Read & truncate safely
    davinci_logprobs = get_logprobs(davinci_file)[:num_tokens]
    ada_logprobs     = get_logprobs(ada_file)[:num_tokens]
    tokens           = get_tokens(davinci_file)[:num_tokens]

    # If lengths mismatch, align to the shortest
    if len(davinci_logprobs) == 0 or len(ada_logprobs) == 0 or len(tokens) == 0:
        # Return a zero feature vector of the right size (7 dims for t_)
        return [0.0]*7
    L = min(len(davinci_logprobs), len(ada_logprobs), len(tokens))
    davinci_logprobs = davinci_logprobs[:L]
    ada_logprobs     = ada_logprobs[:L]
    tokens           = tokens[:L]

    return t_featurize_logprobs(davinci_logprobs, ada_logprobs, tokens)

# ---------- Feature selection ----------

def select_features(exp_to_data, labels, verbose=True, to_normalize=True, indices=None):
    """
    Greedy forward selection over expression features in exp_to_data (dict name -> [N x d_i]).
    Uses k-fold CV score as the objective (k_fold_score).
    """
    if to_normalize:
        normalized_exp_to_data = {}
        for key in exp_to_data:
            normalized_exp_to_data[key] = normalize(exp_to_data[key])
    else:
        normalized_exp_to_data = exp_to_data

    def get_data(*exp):
        return np.concatenate([normalized_exp_to_data[e] for e in exp], axis=1)

    val_exp = list(exp_to_data.keys())
    curr = 0.0
    best_features = []
    i = 0

    iterable = tqdm.tqdm(val_exp) if verbose else val_exp
    while val_exp:
        best_score, best_exp = -1.0, ""
        # Iterate over a *fresh* iterable each round (so tqdm prints properly)
        for exp in (tqdm.tqdm(val_exp) if verbose else val_exp):
            try:
                score = k_fold_score(get_data(*best_features, exp), labels, k=5, indices=indices)
            except Exception:
                # If an expression crashes due to shape/NaN, skip it
                continue
            if score > best_score:
                best_score = score
                best_exp = exp

        if verbose:
            print(f"Iteration {i}, Current Score: {curr:.4f}, Best Feature: {best_exp}, New Score: {best_score:.4f}")

        if best_score <= curr or not best_exp:
            break
        best_features.append(best_exp)
        val_exp.remove(best_exp)
        curr = best_score
        i += 1

    return best_features
