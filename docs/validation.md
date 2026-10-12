# Release Validation

Checks performed on the curated release, not by rerunning model training:

| Check | Result |
|---|---|
| Saved-score metric reproduction | AUC 0.9505208333333334; BAcc 0.8958333333333333; Top-1 0.75; MAE 25 ms |
| CPU regression tests | 5 tests passed, including 5 invalid-table subcases and split leakage |
| Original predictor preservation | AST-identical bodies for all 8 retained functions/classes |
| Architecture-only gradient smoke | Finite/nonzero input gradients using artificial mapper matrices |
| Optional upstream SyncNet inference | One existing prepared clip, CPU, original weights: best offset 0 ms |
| Scalar geometry CLI | Two held-out excerpts evaluated; clearly labeled excerpt-only |
| Plot regeneration | Six plots generated from bundled CSV/JSON only |
| Video composition smoke | Two local clips composed without hardware, source mutation or audio |
| Public media decode | Three MP4s fully decoded; H.264/yuv420p, no audio/subtitle/data streams |
| Local Markdown links | Existing files and heading targets checked |
| Public clone smoke test | GitHub clone completed; the same five tests and saved-score reproduction passed in the downloaded tree |
| Source preservation | 9,669 inventoried source files unchanged in size/mtime; inventoried code/docs and 22 selected sources additionally hashed |
| Private data scan | No credential-pattern, private-path, private-IP or hardware-identifier matches in selected text files |

The CPU checks used Python 3.10.18, NumPy 2.2.6, PyTorch 2.12.0+cu126 (CPU execution),
scikit-learn 1.7.2, Matplotlib 3.10.9, SciPy 1.15.3, OpenCV 4.13.0 and
python_speech_features 0.6 from an existing isolated environment. FFmpeg was used locally.
The requirement files describe compatible ranges; a fresh install on every platform was not tested.

The optional inference smoke uses independently held assets that are not redistributed. It proves
the wrapper runs, not that withheld neural inference can be reproduced from this repository alone.
Media contact sheets were visually checked for people and readable private information; audio was
removed conservatively. Automated scans and visual sampling reduce risk but do not constitute a
legal clearance or a guarantee that every possible privacy issue has been identified.

No GPU training, environment upgrade, robot execution, source-project edit, or original-data deletion
was performed for the public release. Publication status and remote verification are reported
separately; local validation must not be mistaken for a successful GitHub upload.

## Pipeline Figure Update (2026-10-12)

The system overview and Simulator architecture were redrawn from the released model and
protocol notes using `scripts/draw_pipeline.py`. Both figures were visually inspected locally
and exported as PNG, editable-text SVG and vector PDF. The default PNG dimensions are
2880x2178 (overview) and 2880x1998 (architecture).

All **8 CPU regression tests** pass: the original five release tests and three new figure tests.
Figure tests check both text layouts, reject overflowing card text, and verify byte-identical
PNG/SVG/PDF regeneration in the same environment. The historical five-test public-clone record
above describes the initial release; it is not a new clone test for this update.

The figure update changes presentation only. No model, metric table, source experiment,
recording, checkpoint or environment was modified. Hybrid reconstruction and teacher objectives
remain explicitly marked as weak/exploratory, not as verified physical improvements.

## Native README Videos (2026-10-12)

The three existing muted MP4s were attached using GitHub CLI and linked near the top of the
README. Anonymous attachment downloads matched the tracked media SHA256 hashes byte for byte;
all repository originals are H.264 video-only files. No new raw recording or audio was published.

GitHub's official Markdown rendering API returned **three matching `<video controls>` players**,
including the controlled clip inside its collapsed section. Only the stable attachment URLs
are committed, not the temporary signed playback URLs returned in rendered HTML. This checks
server-side player generation and public file access; it is not a manual browser playback test.

All **9 CPU regression tests** pass, including a new attachment/provenance/README-placement check.
Relative links and anchors were checked. Model code, numeric evidence, pipeline figures and
original media files remain unchanged.
