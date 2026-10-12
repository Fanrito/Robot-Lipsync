"""Offline checks binding README players to already-public, immutable media."""

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PublicVideoTests(unittest.TestCase):
    def test_attachment_provenance_and_readme_paragraphs(self):
        manifest = {r["file"]: r for r in json.loads((ROOT / "assets/media_manifest.json").read_text())}
        bindings = json.loads((ROOT / "assets/video_attachments.json").read_text())
        readme = (ROOT / "README.md").read_text()
        rows = bindings["videos"]
        self.assertEqual({row["file"] for row in rows}, set(manifest))
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row["attachment_url"] for row in rows}), 3)
        for row in rows:
            with self.subTest(video=row["file"]):
                path = (ROOT / row["file"]).resolve()
                self.assertTrue(path.is_relative_to(ROOT))
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])
                self.assertEqual(row["sha256"], manifest[row["file"]]["sha256"])
                self.assertRegex(row["attachment_url"],
                                 r"^https://github\.com/user-attachments/assets/[0-9a-f-]{36}$")
                self.assertEqual(readme.count(row["attachment_url"]), 1)
                self.assertIn("\n\n" + row["attachment_url"] + "\n\n", readme)
                self.assertIn(f"]({row['file']})", readme)
        self.assertIn("muted research", readme)
        self.assertLess(readme.index(rows[0]["attachment_url"]), readme.index("## Research Pipeline"))


if __name__ == "__main__":
    unittest.main()
