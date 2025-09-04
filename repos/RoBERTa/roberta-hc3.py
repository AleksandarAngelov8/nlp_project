# save as run_roberta_hc3.py
# usage:
#   python run_roberta_hc3.py
#
# notes:
# - Avoids deprecated HF "dataset scripts" by downloading JSON/JSONL files directly.
# - Forces safetensors for the model so you don't hit the Torch>=2.6 pickle gate.
# - Uses GPU if available.
# - Writes results to OUT (CSV) with p_human, p_ai, and pred columns.

import os
import csv
import sys
from typing import List, Dict

import torch
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# === Config ===
MODEL = "Hello-SimpleAI/chatgpt-detector-roberta"
# Change subset to a smaller domain (e.g., "reddit_eli5", "open_qa", "finance") for quick tests
HC3_SUBSET = "all"

# Output CSV path (adjust if you like)
OUT = r"C:\Users\kucem\PraktikumProject\outputs\roberta\hc3_results.csv"

# Inference params
MAX_LEN = 512
BATCH = 16
USE_SAFETENSORS = True  # keep True to avoid torch.load on .bin


def load_hc3(subset_name: str = "all"):
    """
    Download HC3 data files from the Hub and load with the 'json' builder.
    Tries <subset>.jsonl then <subset>.json.
    Returns a datasets.Dataset (single split).
    """
    for ext in ("jsonl", "json"):
        try:
            fp = hf_hub_download(
                repo_id="Hello-SimpleAI/HC3",
                filename=f"{subset_name}.{ext}",
                repo_type="dataset",
            )
            ds = load_dataset("json", data_files=fp, split="train")
            if len(ds) > 0:
                return ds
        except Exception as e:
            # Try next extension
            last_err = e
            continue
    # If we got here, neither jsonl nor json worked
    raise RuntimeError(
        f"Could not load HC3 subset '{subset_name}' as .jsonl or .json. "
        f"Last error: {repr(last_err)}"
    )


def prepare_rows(ds) -> List[Dict]:
    """
    Flatten HC3 examples into per-answer rows:
      - human answers => label 0
      - chatgpt answers => label 1
    """
    rows = []
    for ex in ds:
        for h in (ex.get("human_answers") or []):
            rows.append({"text": h, "label": 0, "src": "HC3", "kind": "human"})
        for a in (ex.get("chatgpt_answers") or []):
            rows.append({"text": a, "label": 1, "src": "HC3", "kind": "ai"})
    return rows


def main():
    # Load data (robust path; no deprecated scripts)
    ds = load_hc3(HC3_SUBSET)
    rows = prepare_rows(ds)
    if not rows:
        raise RuntimeError("Loaded HC3 dataset has zero flattened rows. Check subset name.")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    # Tokenizer & model (force safetensors to bypass torch.load on .bin)
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    clf = AutoModelForSequenceClassification.from_pretrained(
        MODEL, use_safetensors=USE_SAFETENSORS
    ).to(device).eval()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    # Run batched inference and write CSV
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "p_human", "p_ai", "pred", "src", "kind"])

        for i in tqdm(range(0, len(rows), BATCH), desc="Scoring"):
            batch = rows[i : i + BATCH]
            texts = [r["text"] for r in batch]

            enc = tok(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LEN,
            ).to(device)

            with torch.no_grad():
                logits = clf(**enc).logits
                probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()

            for r, p in zip(batch, probs):
                p_h, p_a = p  # [P(human), P(ai)]
                pred = int(p_a >= 0.5)
                w.writerow(
                    [
                        r["text"],
                        r["label"],
                        f"{p_h:.6f}",
                        f"{p_a:.6f}",
                        pred,
                        r["src"],
                        r["kind"],
                    ]
                )

    print(f"Wrote: {OUT}")
    print(f"Device used: {device}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Surface helpful diagnostics if something goes wrong
        print("ERROR:", repr(e), file=sys.stderr)
        print(
            "Tips:\n"
            "- Ensure internet access for first download (then it caches locally).\n"
            "- If you previously cached a broken HC3 entry, clear it:\n"
            "    huggingface-cli delete-cache --pattern \"Hello-SimpleAI/HC3\"\n"
            "- To run a smaller split quickly, set HC3_SUBSET = 'reddit_eli5' or 'open_qa'.",
            file=sys.stderr,
        )
        sys.exit(1)
