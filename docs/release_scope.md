# Public Release Scope

## Included

- Selected original predictor classes, unchanged in architecture and forward equations.
- Original geometry descriptors and timestamp-aware derivative metric functions.
- Project-written full-sequence SyncNet wrapper; no upstream model copied.
- New validation, plotting and file-only video CLIs around the retained research definitions.
- Small CSV/JSON result tables, scalar trajectory excerpts, source hashes and CPU regression tests.
- Three selected, muted H.264/yuv420p research comparisons with JPEG previews.

## Deliberately Excluded

- Original workspace, datasets, train/optimizer state and all model checkpoints.
- Internal Learn2Talk source, historical archives, meshes, masks and LRS3 assets.
- Robot SDK, blendshape-to-motor mapping implementation, servo ranges, calibration and control scripts.
- Network topology, private addresses, hardware identifiers, credentials and personal account details.
- FLAME/MICA/DECA/EMOCA assets, pretrained mapping arrays and uncertain-license third-party code.
- Human-subject video, voice reference recordings, full lab microphone tracks and textured teacher renders.
- Full training pipelines; this repository does not pretend to release them.

Source identifiers in provenance refer to archived research artifacts, not machine-specific absolute
paths. Git history is initialized for this curated folder only; private workspace history is not imported.
No Git LFS is necessary for the small curated media. No source workspace file is changed by curation.

The public clips show existing renders with their original comparative layout. Audio is removed,
metadata is stripped, and the controlled clip is shortened to its first 24 seconds. No temporal
registration or geometry prediction is changed to improve the demonstration. Display encodings
must not be used to reconstruct command/frame timestamps.

MIT applies only to original source files and associated usage documentation. Media, scientific
results and third-party-derived outputs are not placed under that grant. See [third-party scope](../THIRD_PARTY.md).
