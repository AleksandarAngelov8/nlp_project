# std_eval.py
from __future__ import annotations
import json, math, random, csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Iterable, List, Tuple, Union

@dataclass
class StdRow:
    id: str                 # <dataset>_<split>_<idx>
    dataset: str            # {hc3, ghostbuster, raid, m4}
    domain: str             # e.g., hc3-qa, gb-essay, m4-arxiv, or "mixed"
    detector: str           # e.g., roberta-hc3, ghostbuster-open-gpt2m, binoculars-gpt2m
    y_true: int             # 0=human, 1=ai
    y_score: float          # higher => more AI
    pred: Optional[int]     # optional; None => blank in CSV
    threshold: Optional[float]  # optional; None => blank
    text_len: int           # char length (or token count if you choose)
    meta: str               # JSON-escaped str

CSV_HEADER = ["id","dataset","domain","detector","y_true","y_score","pred","threshold","text_len","meta"]

def json_meta(meta: Dict) -> str:
    """Safe, compact JSON (UTF-8, no newlines)."""
    s = json.dumps(meta, ensure_ascii=False, separators=(",", ":"))
    return s.replace("\n", "\\n").replace("\r", "\\r")

def make_id(dataset: str, split: str, idx: Union[int,str]) -> str:
    return f"{dataset}_{split}_{idx}"

def standardize_score_direction(
    scores: List[float],
    y_true: Optional[List[int]] = None,
    prefer_higher_is_ai: bool = True,
    known_flip: Optional[bool] = None,
) -> Tuple[List[float], bool]:
    """
    Ensure higher = more AI.
    If known_flip is provided, obey it.
    Else, if y_true is available, infer whether we should flip by checking
    the Spearman sign on a small subsample.
    """
    if known_flip is not None:
        flip = known_flip
    else:
        flip = False
        if y_true is not None and len(scores) == len(y_true) and len(scores) > 10:
            # crude sign test: compare mean score for AI vs human
            ai_scores = [s for s, y in zip(scores, y_true) if y == 1]
            hu_scores = [s for s, y in zip(scores, y_true) if y == 0]
            if len(ai_scores) >= 5 and len(hu_scores) >= 5:
                flip = (sum(ai_scores)/len(ai_scores)) < (sum(hu_scores)/len(hu_scores))
    if prefer_higher_is_ai and flip:
        scores = [-s for s in scores]
    return scores, bool(flip)

def apply_threshold(
    y_scores: List[float],
    threshold: Optional[float] = None,
) -> List[Optional[int]]:
    if threshold is None:
        return [None]*len(y_scores)
    return [1 if s >= threshold else 0 for s in y_scores]

def balanced_sample_indexes(
    y_true: List[int], n_per_class: int, rng: random.Random
) -> List[int]:
    idx_h = [i for i, y in enumerate(y_true) if y == 0]
    idx_a = [i for i, y in enumerate(y_true) if y == 1]
    if len(idx_h) < n_per_class or len(idx_a) < n_per_class:
        # If strict enforcement is required, raise; else sample as many as possible.
        raise ValueError(f"Not enough examples to sample {n_per_class} per class "
                         f"(human={len(idx_h)}, ai={len(idx_a)}).")
    rng.shuffle(idx_h); rng.shuffle(idx_a)
    chosen = idx_h[:n_per_class] + idx_a[:n_per_class]
    rng.shuffle(chosen)
    return chosen

def write_unified_csv(
    rows: Iterable[StdRow],
    out_csv: Union[str, Path],
):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for r in rows:
            # None => blank cell for pred/threshold
            w.writerow([
                r.id, r.dataset, r.domain, r.detector, r.y_true, f"{r.y_score:.6f}",
                "" if r.pred is None else int(r.pred),
                "" if r.threshold is None else f"{float(r.threshold):.6f}",
                int(r.text_len),
                r.meta
            ])

class StdCollector:
    """
    Collect raw examples, optionally flip score direction, enforce 3k/3k,
    and write the final CSV.
    """
    def __init__(self, dataset: str, domain: str, detector: str, split: str = "mixed",
                 random_state: int = 42, enforce_3k_per_class: bool = True):
        self.dataset = dataset
        self.domain = domain
        self.detector = detector
        self.split = split
        self.random_state = random_state
        self.enforce_3k = enforce_3k_per_class
        self._tmp: List[Tuple[str,int,float,Optional[int],Optional[float],int,str]] = []
        # fields: (id, y_true, y_score, pred, thr, text_len, meta_json)

    def add(
        self, *,
        idx: Union[int,str],
        y_true: int,
        y_score: float,
        text_len: int,
        pred: Optional[int] = None,
        threshold: Optional[float] = None,
        meta: Optional[Dict] = None,
    ):
        rid = make_id(self.dataset, self.split, idx)
        m = json_meta(meta or {})
        self._tmp.append((rid, y_true, float(y_score), pred, threshold, int(text_len), m))

    def finalize(
        self,
        out_csv: Union[str, Path],
        known_flip: Optional[bool] = None,
        threshold: Optional[float] = None,   # if you want to compute pred here
    ):
        rng = random.Random(self.random_state)
        ids, ys, scores, preds, thrs, lens, metas = zip(*self._tmp) if self._tmp else ([],[],[],[],[],[],[])

        # 1) Standardize direction (higher=AI)
        scores, flipped = standardize_score_direction(list(scores), list(ys), known_flip=known_flip)

        # 2) Optional pred computation at a fixed threshold (leave None to keep blank)
        if threshold is not None:
            preds = apply_threshold(scores, threshold)
            thrs = [threshold]*len(scores)

        # 3) Enforce 3k/3k sample (balanced) if requested
        chosen_idx = list(range(len(scores)))
        if self.enforce_3k:
            chosen_idx = balanced_sample_indexes(list(ys), 3000, rng)

        # 4) Build StdRow list in the required column order
        rows: List[StdRow] = []
        for i in chosen_idx:
            rows.append(StdRow(
                id=ids[i],
                dataset=self.dataset,
                domain=self.domain,
                detector=self.detector,
                y_true=int(ys[i]),
                y_score=float(scores[i]),
                pred=preds[i] if preds else None,
                threshold=thrs[i] if thrs else None,
                text_len=int(lens[i]),
                meta=metas[i],
            ))

        # 5) Write CSV
        write_unified_csv(rows, out_csv)

        return {
            "n_total": len(self._tmp),
            "n_written": len(rows),
            "flipped": flipped,
            "threshold": threshold,
            "out": str(out_csv),
        }
