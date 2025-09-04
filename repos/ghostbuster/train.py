import argparse
import math
import time
import numpy as np
import tiktoken
import dill as pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

from tabulate import tabulate
from tqdm.auto import tqdm

from utils.featurize import normalize, t_featurize, select_features
from utils.symbolic import get_all_logprobs, train_trigram, get_exp_featurize
from utils.symbolic import generate_symbolic_data
from utils.load import get_generate_dataset, Dataset


# ----------------- small helpers -----------------
def _as_file_list(generate_dataset_fn):
    """Get the ordered list of file paths in the dataset pipeline."""
    files = generate_dataset_fn(lambda f: f)
    # ensure plain python list of strings
    if hasattr(files, "tolist"):
        files = files.tolist()
    return list(files)


def _build_matrix_with_progress(files, featurize_fn, desc):
    """
    Apply featurize_fn(file) over files with a progress bar.
    Returns np.array of shape [N, D].
    """
    rows = []
    t0 = time.time()
    for f in tqdm(files, desc=desc, unit="file"):
        rows.append(featurize_fn(f))
    dt = time.time() - t0
    print(f"[info] {desc}: built {len(rows)} rows in {dt:0.1f}s")
    # robust stack
    X = np.array(rows)
    if X.dtype == object:
        X = np.stack(rows, axis=0)
    return X


# ----------------- datasets -----------------
with open("results/best_features_four.txt", encoding="utf-8") as f:
    best_features = f.read().strip().split("\n")

print("Loading trigram model...")
trigram_model = pickle.load(open("model/trigram_model.pkl", "rb"), pickle.HIGHEST_PROTOCOL)
tokenizer = tiktoken.encoding_for_model("davinci").encode

wp_dataset = [
    Dataset("normal", "data/wp/human"),
    Dataset("normal", "data/wp/gpt"),
]

reuter_dataset = [
    Dataset("author", "data/reuter/human"),
    Dataset("author", "data/reuter/gpt"),
]

essay_dataset = [
    Dataset("normal", "data/essay/human"),
    Dataset("normal", "data/essay/gpt"),
]

eval_dataset = [
    Dataset("normal", "data/wp/claude"),
    Dataset("author", "data/reuter/claude"),
    Dataset("normal", "data/essay/claude"),
    Dataset("normal", "data/wp/gpt_prompt1"),
    Dataset("author", "data/reuter/gpt_prompt1"),
    Dataset("normal", "data/essay/gpt_prompt1"),
    Dataset("normal", "data/wp/gpt_prompt2"),
    Dataset("author", "data/reuter/gpt_prompt2"),
    Dataset("normal", "data/essay/gpt_prompt2"),
    Dataset("normal", "data/wp/gpt_writing"),
    Dataset("author", "data/reuter/gpt_writing"),
    Dataset("normal", "data/essay/gpt_writing"),
    Dataset("normal", "data/wp/gpt_semantic"),
    Dataset("author", "data/reuter/gpt_semantic"),
    Dataset("normal", "data/essay/gpt_semantic"),
]


# ----------------- feature builder (now with progress) -----------------
def get_featurized_data(generate_dataset_fn, best_features):
    # 1) enumerate files once
    files = _as_file_list(generate_dataset_fn)
    print(f"[info] Found {len(files)} files in dataset pipeline")

    # 2) classic t_ features with a visible progress bar
    t_data = _build_matrix_with_progress(files, t_featurize, desc="t_features")

    # 3) load vector streams (davinci/ada/trigram/unigram) – print timing
    print("[info] building logprob streams (davinci/ada/trigram/unigram)...")
    t0 = time.time()
    davinci, ada, trigram, unigram = get_all_logprobs(
        generate_dataset_fn, trigram=trigram_model, tokenizer=tokenizer
    )
    print(f"[info] logprob streams ready in {time.time()-t0:0.1f}s")

    # 4) symbolic expressions → vector function
    vector_map = {
        "davinci-logprobs": lambda file: davinci[file],
        "ada-logprobs": lambda file: ada[file],
        "trigram-logprobs": lambda file: trigram[file],
        "unigram-logprobs": lambda file: unigram[file],
    }
    exp_featurize = get_exp_featurize(best_features, vector_map)

    # 5) build exp_data with progress
    exp_data = _build_matrix_with_progress(files, exp_featurize, desc="exp_features")

    # 6) concat
    X = np.concatenate([t_data, exp_data], axis=1)
    print(f"[info] feature matrix: {X.shape[0]} rows × {X.shape[1]} dims")
    return X


# ----------------- main -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_symbolic_data", action="store_true")
    parser.add_argument("--generate_symbolic_data_four", action="store_true")
    parser.add_argument("--generate_symbolic_data_eval", action="store_true")

    parser.add_argument("--perform_feature_selection", action="store_true")
    parser.add_argument("--perform_feature_selection_one", action="store_true")
    parser.add_argument("--perform_feature_selection_two", action="store_true")
    parser.add_argument("--perform_feature_selection_four", action="store_true")

    parser.add_argument("--perform_feature_selection_only_ada", action="store_true")
    parser.add_argument("--perform_feature_selection_no_gpt", action="store_true")
    parser.add_argument("--perform_feature_selection_domain", action="store_true")

    parser.add_argument("--only_include_gpt", action="store_true")
    parser.add_argument("--train_on_all_data", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    result_table = [["F1", "Accuracy", "AUC"]]

    datasets = [
        *wp_dataset,
        *reuter_dataset,
        *essay_dataset,
    ]
    generate_dataset_fn = get_generate_dataset(*datasets)

    # Optional generation stages (they already print; we add small banners)
    if args.generate_symbolic_data:
        print("[stage] generate_symbolic_data (depth=3)")
        generate_symbolic_data(
            generate_dataset_fn,
            max_depth=3,
            output_file="symbolic_data_gpt",
            verbose=True,
        )
        t_data = generate_dataset_fn(t_featurize)
        pickle.dump(t_data, open("t_data", "wb"))

    if args.generate_symbolic_data_eval:
        print("[stage] generate_symbolic_data_eval (depth=3)")
        generate_dataset_fn_eval = get_generate_dataset(*eval_dataset)
        generate_symbolic_data(
            generate_dataset_fn_eval,
            max_depth=3,
            output_file="symbolic_data_eval",
            verbose=True,
        )
        t_data_eval = generate_dataset_fn_eval(t_featurize)
        pickle.dump(t_data_eval, open("t_data_eval", "wb"))

    if args.generate_symbolic_data_four:
        print("[stage] generate_symbolic_data_four (depth=4)")
        generate_symbolic_data(
            generate_dataset_fn,
            max_depth=4,
            output_file="symbolic_data_gpt_four",
            verbose=True,
        )
        t_data = generate_dataset_fn(t_featurize)
        pickle.dump(t_data, open("t_data", "wb"))

    # Labels + split (unchanged)
    labels = generate_dataset_fn(lambda file: 1 if any([m in file for m in ["gpt", "claude"]]) else 0)

    indices = np.arange(len(labels))
    if args.only_include_gpt:
        where_gpt = np.where(generate_dataset_fn(lambda file: 0 if "claude" in file else 1))[0]
        indices = indices[where_gpt]

    np.random.shuffle(indices)
    train, test = (
        indices[: math.floor(0.8 * len(indices))],
        indices[math.floor(0.8 * len(indices)) :],
    )
    print("Train/Test Split", train, test)
    print("Train Size:", len(train), "Valid Size:", len(test))
    print(f"Positive Labels: {sum(labels[indices])}, Total Labels: {len(indices)}")

    # Feature selection stages already use tqdm inside select_features()
    if args.perform_feature_selection:
        print("[stage] perform_feature_selection (depth=3 cache)")
        exp_to_data = pickle.load(open("symbolic_data_gpt", "rb"))
        best_features = select_features(exp_to_data, labels, verbose=True, to_normalize=True, indices=train)
        with open("results/best_features_three.txt", "w", encoding="utf-8") as f:
            for feat in best_features:
                f.write(feat + "\n")

    if args.perform_feature_selection_two:
        print("[stage] perform_feature_selection_two (≤4 tokens)")
        old_exp_to_data = pickle.load(open("symbolic_data_gpt", "rb"))
        exp_to_data = {k: v for k, v in old_exp_to_data.items() if len(k.split(" ")) <= 4}
        best_features = select_features(exp_to_data, labels, verbose=True, to_normalize=True, indices=train)
        with open("results/best_features_two.txt", "w", encoding="utf-8") as f:
            for feat in best_features:
                f.write(feat + "\n")

    if args.perform_feature_selection_one:
        print("[stage] perform_feature_selection_one (≤2 tokens)")
        old_exp_to_data = pickle.load(open("symbolic_data_gpt", "rb"))
        exp_to_data = {k: v for k, v in old_exp_to_data.items() if len(k.split(" ")) <= 2}
        best_features = select_features(exp_to_data, labels, verbose=True, to_normalize=True, indices=train)
        with open("results/best_features_one.txt", "w", encoding="utf-8") as f:
            for feat in best_features:
                f.write(feat + "\n")

    if args.perform_feature_selection_four:
        print("[stage] perform_feature_selection_four (depth=4 cache)")
        exp_to_data = pickle.load(open("symbolic_data_gpt_four", "rb"))
        best_features = select_features(exp_to_data, labels, verbose=True, to_normalize=True, indices=train)
        with open("results/best_features_four.txt", "w", encoding="utf-8") as f:
            for feat in best_features:
                f.write(feat + "\n")

    if args.perform_feature_selection_no_gpt:
        print("[stage] perform_feature_selection_no_gpt (drop ada/davinci)")
        old_exp_to_data = pickle.load(open("symbolic_data_gpt", "rb"))
        exp_to_data = {k: v for k, v in old_exp_to_data.items() if ("ada" not in k and "davinci" not in k)}
        best_features = select_features(exp_to_data, labels, verbose=True, to_normalize=True, indices=train)
        with open("results/best_features_no_gpt.txt", "w", encoding="utf-8") as f:
            for feat in best_features:
                f.write(feat + "\n")

    if args.perform_feature_selection_only_ada:
        print("[stage] perform_feature_selection_only_ada (drop davinci)")
        old_exp_to_data = pickle.load(open("symbolic_data_gpt", "rb"))
        exp_to_data = {k: v for k, v in old_exp_to_data.items() if "davinci" not in k}
        best_features = select_features(exp_to_data, labels, verbose=True, to_normalize=True, indices=train)
        with open("results/best_features_only_ada.txt", "w", encoding="utf-8") as f:
            for feat in best_features:
                f.write(feat + "\n")

    if args.perform_feature_selection_domain:
        print("[stage] perform_feature_selection_domain (per-domain)")
        exp_to_data = pickle.load(open("symbolic_data_gpt", "rb"))
        wp_indices = np.where(generate_dataset_fn(lambda file: "wp" in file))[0]
        reuter_indices = np.where(generate_dataset_fn(lambda file: "reuter" in file))[0]
        essay_indices = np.where(generate_dataset_fn(lambda file: "essay" in file))[0]

        wp_features = select_features(exp_to_data, labels, verbose=True, to_normalize=True, indices=wp_indices)
        with open("results/best_features_wp.txt", "w", encoding="utf-8") as f:
            for feat in wp_features:
                f.write(feat + "\n")

        reuter_features = select_features(exp_to_data, labels, verbose=True, to_normalize=True, indices=reuter_indices)
        with open("results/best_features_reuter.txt", "w", encoding="utf-8") as f:
            for feat in reuter_features:
                f.write(feat + "\n")

        essay_features = select_features(exp_to_data, labels, verbose=True, to_normalize=True, indices=essay_indices)
        with open("results/best_features_essay.txt", "w", encoding="utf-8") as f:
            for feat in essay_features:
                f.write(feat + "\n")

    # ---- build full feature matrix with progress ----
    print("[stage] building feature matrix (t_ + exp features)")
    data_raw = get_featurized_data(generate_dataset_fn, best_features)

    # ---- normalization ----
    print("[stage] normalizing features")
    data, mu, sigma = normalize(data_raw, ret_mu_sigma=True)
    print(f"Best Features: {best_features}")
    print(f"Data Shape: {data.shape}")

    # ---- model ----
    print("[stage] training calibrated logistic regression")
    base = LogisticRegression()
    model = CalibratedClassifierCV(base, cv=5)

    if args.train_on_all_data:
        model.fit(data, labels)

        with open("model/features.txt", "w", encoding="utf-8") as f:
            for feat in best_features:
                f.write(feat + "\n")
        pickle.dump(model, open("model/model", "wb"))
        pickle.dump(mu, open("model/mu", "wb"))
        pickle.dump(sigma, open("model/sigma", "wb"))
        print("Saved model to model/")
    else:
        model.fit(data[train], labels[train])

    print("[stage] evaluating holdout split")
    predictions = model.predict(data[test])
    probs = model.predict_proba(data[test])[:, 1]

    result_table.append(
        [
            round(f1_score(labels[test], predictions), 3),
            round(accuracy_score(labels[test], predictions), 3),
            round(roc_auc_score(labels[test], probs), 3),
        ]
    )

    print(tabulate(result_table, headers="firstrow", tablefmt="grid"))
