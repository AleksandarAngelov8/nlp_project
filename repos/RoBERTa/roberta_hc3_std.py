# save as run_roberta_hc3.py
# usage:
#   python run_roberta_hc3.py
#
# Emits unified CSV schema:
# id,dataset,domain,detector,y_true,y_score,pred,threshold,text_len,meta
#
# Requires std_eval.py (from previous message) in PYTHONPATH or same directory.

import os
import sys
from typing import List, Dict

import torch
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- NEW: unified collector ---
from std_eval import StdCollector  # ensure std_eval.py is available

# === Config ===
MODEL = "Hello-SimpleAI/chatgpt-detector-roberta"
HC3_SUBSET = "all"   # e.g., "reddit_eli5", "open_qa", "finance", or "all"

# Unified CSV output
OUT = r"C:\Users\kucem\PraktikumProject\outputs\unified\hc3_roberta_hc3-all.csv"

# Inference params
MAX_LEN = 512
BATCH = 16
USE_SAFETENSORS = False  # keep True to avoid torch.load on .bin

DATASET_TAG = "hc3"
DOMAIN_TAG = f"hc3-{HC3_SUBSET}"  # example: "hc3-all"
DETECTOR_TAG = "roberta-hc3"      # adjust if desired
SPLIT_TAG = HC3_SUBSET            # used inside id construction (via StdCollector)

def load_hc3(subset_name: str = "all"):
    """
    Download HC3 data files from the Hub and load with the 'json' builder.
    Tries <subset>.jsonl then <subset>.json.
    Returns a datasets.Dataset (single split).
    """
    last_err = None
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
            last_err = e
            continue
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
        # Best-effort stable reference if present
        qid = ex.get("question_id") or ex.get("id")
        # Collect humans
        for h in (ex.get("human_answers") or []):
            rows.append({
                "text": h,
                "label": 0,
                "src": "HC3",
                "kind": "human",
                "qid": qid
            })
        # Collect AI
        for a in (ex.get("chatgpt_answers") or []):
            rows.append({
                "text": a,
                "label": 1,
                "src": "HC3",
                "kind": "ai",
                "qid": qid
            })
    return rows

def main():
    # Load data
    ds = load_hc3(HC3_SUBSET)
    rows = prepare_rows(ds)
    if not rows:
        raise RuntimeError("Loaded HC3 dataset has zero flattened rows. Check subset name.")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    # Tokenizer & model
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    clf = AutoModelForSequenceClassification.from_pretrained(
        MODEL, use_safetensors=USE_SAFETENSORS
    ).to(device).eval()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    # Unified collector (enforces 3k human + 3k AI)
    collector = StdCollector(
        dataset=DATASET_TAG,
        domain=DOMAIN_TAG,
        detector=DETECTOR_TAG,
        split=SPLIT_TAG,
        random_state=42,
        enforce_3k_per_class=True
    )

    # Batched inference
    idx_counter = 0  # stable within this run/order
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
            p_h, p_ai = p  # [P(human), P(ai)]
            y_true = int(r["label"])         # 0 human, 1 ai
            y_score = float(p_ai)            # higher => more AI
            text_len = len(r["text"])
            meta = {
                "source": r.get("src"),
                "subset": HC3_SUBSET,
                "kind": r.get("kind"),
                "qid": r.get("qid"),
                "model": MODEL,
                "max_len": MAX_LEN
            }
            collector.add(
                idx=idx_counter,
                y_true=y_true,
                y_score=y_score,
                text_len=text_len,
                meta=meta
                # pred=None, threshold=None -> left blank per unified schema
            )
            idx_counter += 1

    # Finalize unified CSV
    # known_flip=False since p_ai already means "higher => more AI"
    info = collector.finalize(
        out_csv=OUT,
        known_flip=False,
        threshold=None  # leave pred/threshold blank in unified export
    )

    print(f"[OK] Unified CSV written: {info['out']}")
    print(f"  - n_total collected: {info['n_total']}")
    print(f"  - n_written: {info['n_written']} (should be 6000 = 3k+3k)")
    print(f"  - flipped: {info['flipped']}")
    print(f"Device used: {device}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
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
