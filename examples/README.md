# Available Examples

The main README's metric example is self-contained: the two curve CSVs contain the real saved
sequence-level validation/test predictions, not fabricated data. The descriptor excerpt contains
two real held-out scalar trajectories, not the full 3D motion dataset.

`scripts/smoke_model.py` is deliberately different: random matrices exercise tensor shape and
gradient flow without distributing pretrained mapper arrays. Its printed success is a software
test, not evidence of motion quality.

For user-supplied geometry, `scripts/evaluate_flame.py --meshes example.npz --output outputs/mesh.json`
expects `prediction` and `target` with shape `[T,V,3]` in the same coordinates, plus unique integer
`mouth_indices [M]`. The caller must align timestamps and obtain the necessary data/mask rights.
For scalar excerpts, see `results/speech_v2_descriptor_excerpt.csv` for the exact schema.

For review composition (requires FFmpeg and ffprobe):

```bash
python scripts/make_side_by_side.py \
  --inputs assets/videos/speech_natural.mp4 assets/videos/speech_bilabial.mp4 \
  --output outputs/video_tool_smoke.mp4
```

This command is only a composition smoke test: those clips are **different utterances**, so the
result is not an aligned scientific comparison. For actual A/B review supply already time-aligned
clips of the same utterance. The tool pads tiles without aspect distortion, stops at the shorter
input, removes audio, and refuses to overwrite an existing output or source file.
