#!/usr/bin/env python3
"""Evaluate released scalar excerpts, or user-provided weak-target mesh arrays."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.geometry import derivative_errors, robust_span, safe_corr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--curves", type=Path)
    source.add_argument("--meshes", type=Path, help="NPZ: prediction,target [T,V,3], mouth_indices [M]")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.curves:
        with args.curves.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("Empty scalar curve table")
        report = {"scope": "Selected excerpts only; not the full test-set result", "samples": {}}
        for sample in sorted({r["sample_id"] for r in rows}):
            group = [r for r in rows if r["sample_id"] == sample]
            times = np.asarray([float(r["time_sec"]) for r in group])
            if len(times) < 3 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError("Need at least three strictly increasing finite timestamps per clip")
            metrics = {}
            for key in ("jaw", "inner", "outer", "width", "asymmetry"):
                p = np.asarray([float(r[key+"_pred"]) for r in group])
                t = np.asarray([float(r[key+"_target"]) for r in group])
                if not np.isfinite(p).all() or not np.isfinite(t).all():
                    raise ValueError("Non-finite descriptor")
                metrics[key] = dict(correlation=safe_corr(p, t), mae=float(np.mean(np.abs(p-t))),
                    amplitude_ratio=robust_span(p)/max(robust_span(t), 1e-8),
                    **derivative_errors(p[:, None], t[:, None], np.rint(times*1e9).astype(np.int64), np.zeros(len(times))))
            report["samples"][sample] = metrics
    else:
        with np.load(args.meshes, allow_pickle=False) as data:
            pred, target, indices = data["prediction"], data["target"], data["mouth_indices"]
        if pred.shape != target.shape or pred.ndim != 3 or pred.shape[-1] != 3 or not pred.size:
            raise ValueError("Meshes must have identical nonempty [T,V,3] shapes")
        if not np.isfinite(pred).all() or not np.isfinite(target).all():
            raise ValueError("Non-finite vertices")
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices) or indices.min() < 0 or indices.max() >= pred.shape[1] or len(np.unique(indices)) != len(indices):
            raise ValueError("Invalid or duplicate mouth indices")
        error = np.abs(pred-target)
        report = dict(frames=len(pred), full_vertex_mae=float(error.mean()), mouth_vertex_mae=float(error[:, indices].mean()),
                      definition="Mean absolute coordinate error, not mean Euclidean vertex distance; input coordinate units")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
