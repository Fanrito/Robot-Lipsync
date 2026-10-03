"""Small CPU regression tests; no dataset, pretrained weight or robot needed."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.synchronization import evaluate_splits, read_curves

ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def test_recorded_full_sequence_metrics(self):
        v = read_curves(ROOT / "results/sync2d_val_curves.csv")
        t = read_curves(ROOT / "results/sync2d_test_curves.csv")
        result = evaluate_splits(v, t)
        self.assertAlmostEqual(result["auc"], .9505208333333334)
        self.assertAlmostEqual(result["balanced_accuracy"], .8958333333333333)
        self.assertEqual(result["zero_top1"], .75)
        self.assertEqual(result["offset_mae_ms"], 25.)

    def test_leakage_fails(self):
        rows = read_curves(ROOT / "results/sync2d_test_curves.csv")
        with self.assertRaisesRegex(ValueError, "leakage"):
            evaluate_splits(rows, rows)

    def test_bad_curves_fail(self):
        with (ROOT / "results/sync2d_test_curves.csv").open() as f:
            original = list(csv.DictReader(f))
        for kind in ("nan", "duplicate", "bad_label", "bad_score", "missing_offset"):
            rows = [r.copy() for r in original]
            if kind == "nan": rows[0]["mean_distance"] = "nan"
            if kind == "duplicate": rows.append(rows[0])
            if kind == "bad_label": rows[0]["label"] = "1"
            if kind == "bad_score": rows[0]["relative_score"] = "123"
            if kind == "missing_offset": rows.pop(0)
            with tempfile.TemporaryDirectory() as d:
                path = Path(d)/"curves.csv"
                with path.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    read_curves(path)

    def test_descriptor_metrics(self):
        from src.evaluation.geometry import robust_span, safe_corr, derivative_errors
        y = np.linspace(0, 1, 10)
        self.assertAlmostEqual(robust_span(y), .96)
        self.assertAlmostEqual(safe_corr(y, y), 1)
        self.assertIsNone(safe_corr(y*0, y))
        t = np.arange(10, dtype=np.int64)*40_000_000
        result = derivative_errors(y[:, None], y[:, None], t, np.zeros(10))
        self.assertEqual(result["velocity_rmse"], 0)
        self.assertEqual(result["acceleration_rmse"], 0)

    def test_schema_name_order(self):
        import torch
        from src.robot_flame.model import NoetixBlendshapeAdapter, NOETIX_51_NAMES, MEDIAPIPE_52_NAMES, validate_noetix_schema
        adapter = NoetixBlendshapeAdapter()
        x = torch.zeros(1, 51); x[0, NOETIX_51_NAMES.index("JawOpen")] = .7
        result = adapter.to_mediapipe52(x)
        self.assertAlmostEqual(result[0, MEDIAPIPE_52_NAMES.index("JawOpen")].item(), .7, places=6)
        self.assertAlmostEqual(result[0, 0].item(), .3, places=6)
        self.assertEqual(adapter.to_arkit52(x)[0, -1], 0)
        with self.assertRaises(ValueError):
            validate_noetix_schema(tuple(reversed(NOETIX_51_NAMES)))


if __name__ == "__main__":
    unittest.main()
