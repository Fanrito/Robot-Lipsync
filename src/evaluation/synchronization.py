"""Sequence-curve metrics with an explicit validation/test boundary."""
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


def read_curves(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Empty curve table")
    seen, groups = set(), defaultdict(list)
    for row in rows:
        for field in ("offset_ms", "mean_distance", "relative_score"):
            row[field] = float(row[field])
            if not np.isfinite(row[field]):
                raise ValueError(f"Non-finite {field}")
        row["label"] = int(row["label"])
        if row["label"] != int(row["offset_ms"] == 0):
            raise ValueError("This CLI evaluates zero-offset positives only")
        key = (row["sample_id"], row["offset_ms"])
        if key in seen:
            raise ValueError("Expected one aggregate row per utterance and offset")
        seen.add(key)
        groups[row["sample_id"]].append(row)
    reference_offsets = None
    for sample in groups.values():
        offsets = {r["offset_ms"] for r in sample}
        if 0 not in offsets or len(offsets) < 2:
            raise ValueError("Each utterance must include zero and nonzero offsets")
        if reference_offsets is not None and reference_offsets != offsets:
            raise ValueError("Utterances must use the same offset grid")
        reference_offsets = offsets
        if len({r["category"] for r in sample}) != 1:
            raise ValueError("One utterance cannot have multiple categories")
        median = np.median([r["mean_distance"] for r in sample])
        if not all(np.isclose(r["relative_score"], median-r["mean_distance"], atol=1e-6) for r in sample):
            raise ValueError("relative_score must equal median(curve) minus distance")
    return rows


def calibrate_threshold(rows: list[dict]) -> tuple[float, float]:
    """Match the historical validation-only unique-score threshold search."""
    labels = np.asarray([r["label"] for r in rows])
    scores = np.asarray([r["relative_score"] for r in rows])
    if set(labels) != {0, 1}:
        raise ValueError("Calibration requires both classes")
    best = (float(np.median(scores)), -1.0)
    for threshold in np.unique(scores):
        value = balanced_accuracy_score(labels, scores >= threshold)
        if value > best[1]:
            best = float(threshold), float(value)
    return best


def evaluate_curves(rows: list[dict], threshold: float) -> dict:
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    y = np.asarray([r["label"] for r in rows])
    scores = np.asarray([r["relative_score"] for r in rows])
    if set(y) != {0, 1}:
        raise ValueError("Evaluation requires both classes")
    groups = defaultdict(list)
    for row in rows:
        groups[row["sample_id"]].append(row)
    selected, ties = [], 0
    for sample in groups.values():
        # Historical argmin behavior: first minimum in sorted offset order.
        sample = sorted(sample, key=lambda r: r["offset_ms"])
        best = min(sample, key=lambda r: r["mean_distance"])
        ties += sum(r["mean_distance"] == best["mean_distance"] for r in sample) > 1
        selected.append(dict(sample_id=best["sample_id"], offset_ms=best["offset_ms"]))
    return dict(pairs=len(rows), samples=len(groups), threshold=threshold,
                auc=float(roc_auc_score(y, scores)),
                accuracy=float(accuracy_score(y, scores >= threshold)),
                balanced_accuracy=float(balanced_accuracy_score(y, scores >= threshold)),
                zero_top1=float(np.mean([r["offset_ms"] == 0 for r in selected])),
                offset_mae_ms=float(np.mean([abs(r["offset_ms"]) for r in selected])),
                exact_tie_utterances=ties, selected_offsets=selected)


def evaluate_splits(validation: list[dict], test: list[dict]) -> dict:
    if {r["sample_id"] for r in validation} & {r["sample_id"] for r in test}:
        raise ValueError("Validation/test utterance leakage")
    threshold, val_bacc = calibrate_threshold(validation)
    result = evaluate_curves(test, threshold)
    result["validation_balanced_accuracy"] = val_bacc
    result["category_metrics"] = {
        cat: evaluate_curves([r for r in test if r["category"] == cat], threshold)
        for cat in sorted({r["category"] for r in test})
    }
    return result
