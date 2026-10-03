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
