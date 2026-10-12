"""Offline regression checks for public research figures."""

from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.draw_pipeline import Diagram, overview, simulator_detail


class PipelineFigureTests(unittest.TestCase):
    def test_both_layouts_fit(self):
        for build in (overview, simulator_detail):
            with self.subTest(figure=build.__name__):
                diagram = build()
                try:
                    diagram.validate_text()
                finally:
                    plt.close(diagram.fig)

    def test_invalid_card_text_is_rejected(self):
        diagram = Diagram(200, 100)
        try:
            diagram.text(50, 50, "text outside small card", region=(50, 40, 5, 20))
            with self.assertRaisesRegex(ValueError, "Text outside"):
                diagram.validate_text()
        finally:
            plt.close(diagram.fig)

    def test_export_formats_and_repeatability(self):
        with tempfile.TemporaryDirectory() as directory, matplotlib.rc_context({
            "svg.fonttype": "none", "svg.hashsalt": "robot-lipsync-pipeline-v1", "pdf.fonttype": 42,
        }):
            root = Path(directory)
            overview().save(root / "first", "overview", dpi=72)
            overview().save(root / "second", "overview", dpi=72)
            for extension in ("png", "svg", "pdf"):
                first = (root / "first" / f"overview.{extension}").read_bytes()
                second = (root / "second" / f"overview.{extension}").read_bytes()
                self.assertEqual(first, second, extension)
            svg = ET.parse(root / "first/overview.svg")
            for line in (root / "first/overview.svg").read_text().splitlines():
                self.assertEqual(line, line.rstrip())
            text = " ".join(svg.getroot().itertext())
            self.assertIn("not measured 3D ground truth", text)
            self.assertIn("LatentSync stays in 2D", text)
            self.assertTrue((root / "first/overview.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertTrue((root / "first/overview.pdf").read_bytes().startswith(b"%PDF-"))


if __name__ == "__main__":
    unittest.main()
