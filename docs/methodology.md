# Protocol and Implementation Notes

## 2D Sequence Evaluation

The released `src/syncnet/sequence.py` functions are project-written wrappers selected from the
existing sequence experiment. Their bodies are unchanged. The upstream neural architecture is
not copied. Prepared input is 25-Hz BGR video at 224x224 and 16-kHz mono int16 audio. Five frames
form a visual feature window; `python_speech_features.mfcc` provides a 13x20 audio window.

For this historical **2D** wrapper, positive offset means:

```text
compare video(t) with audio(t + offset_ms / 1000)
```

Each offset uses all its own valid overlapping windows. Edge-window counts can differ across
offsets; the published `window_count` column makes this explicit. Mean embedding L2 distance
is calculated over the sequence. The sync score is `median(offset distances) - distance(offset)`.
This is not the original per-shift confidence used in the earlier 54-clip run.

The released evaluator rejects missing offsets, non-finite data, duplicate sequence-offset rows,
inconsistent labels and validation/test utterance overlap. It chooses the binary threshold on
validation only, preserving the original unique-score search and ascending-offset tie behavior.
Reported test AUC treats each sequence-offset pair as an observation; those pairs are correlated,
not 56 independent recordings. No significance claim or confidence interval is inferred.

Full-frame images retain aspect ratio and are letterboxed to 224x224. The upstream spatial
protocol instead uses S3FD tracks, median-smoothed boxes and `crop_scale=0.40`. Project square
face crops and mouth crops are separate ablations. None may be silently substituted for another.

### Optional 2D Inference

Obtain [upstream SyncNet](https://github.com/joonson/syncnet_python) under its own terms.
The historical source commit was `907c0b579c2e2d83f0eae1b2ac9e720cde4e5623`.
Download weights following the upstream instructions. They are not supplied here.
Prepare input with the intended spatial protocol before running:

```bash
python scripts/infer_sync.py \
  --syncnet-dir external/syncnet_python \
  --checkpoint external/syncnet_v2.model \
  --video examples/my_prepared_25hz_224.mp4 \
  --audio examples/my_audio_16k_mono.wav \
  --spatial-protocol full_frame \
  --sample-id example01 --category natural \
  --device cpu --output outputs/example01_curve.csv
```

These example inputs must be provided by the user. Model loading uses restricted tensor loading
and strict keys, allowing only absent historical BatchNorm integer counters to be initialized.
It does not fall back to unrestricted pickle loading. The original wrapper materializes video
windows in memory: use short clips, not long sessions. CUDA is optional, not assumed.

## Blendshape-Conditioned Simulator

`src/robot_flame/model.py` contains the original project-owned predictor and schema classes,
without the environment-specific DECA loader or FLAME decoder. Network bodies and state keys
are retained, including the symmetric-padded TCN over a finite past-command history; it is not
claimed to be a redesigned streaming causal convolution.

The name-based adapter first adds `TongueOut=0`, then gathers the 51 shared channels into the
mapper's MediaPipe order and uses `_neutral = clamp(1-max(shared),0,1)`. Only that neutral
category is clamped: **input blendshapes are not clipped** to the old controlled-domain range.
The converter's documented ARKit52 order is Noetix order plus TongueOut, not a universal ARKit
serialization order. Call `validate_noetix_schema` before supplying a recorded tensor.

The external mapper was pinned to
`PeizhiYan/mediapipe-blendshapes-to-flame@34c4964cdb9516540900a556dba1a71493f07138`:
`bs2exp.npy [52,50]`, `bs2pose.npy [52,6]`; only the last three pose coefficients are jaw.
The mapper is frozen. Its later 100D-expression variants are not interchangeable.

The original predictor supports the static prior, static residual, temporal-from-scratch,
static+temporal, affine/MLP adapter, and adapter+temporal variants. Defaults are hidden size 192,
four residual TCN blocks (dilations 1/2/4/8), LayerNorm/GELU/dropout, phase features
`tau, sin(2*pi*tau), cos(2*pi*tau)`, 50 expression and 3 jaw outputs. Residual heads start at zero.
The downstream FLAME decoder uses one fixed identity and canonical global pose. Decoder assets,
identity coefficients and checkpoints are withheld; no placeholder geometry is substituted.

For timestamp preparation, actual 10-Hz command indices and camera monotonic timestamps determine
the history and interval phase. Repeated camera PTS retain the first observation; intervals must
not bridge utterance/split boundaries. Do not infer the camera timeline from `frame_index / 30` or
guess the 30-to-10-Hz selection phase. Predictor inputs have shape `[batch,12,51]` and `[batch]`.
Normalization statistics must come from training only. `scripts/smoke_model.py` tests this interface
with artificial matrices and asserts finite nonzero gradients; it cannot reproduce pretrained accuracy.

## Geometry Metrics

The extracted functions in `src/evaluation/geometry.py` retain the historical definitions:

- 68 landmarks are centered at the mean of indices 0 and 16 and scaled by their 2D distance.
- Inner opening is the Euclidean distance 62-to-66; outer opening is 51-to-57.
- Width is 48-to-54; asymmetry is the absolute y difference of 48 and 54.
- Correlation is Pearson correlation, undefined for degenerate/constant trajectories.
- Amplitude ratio uses `P98-P2`, prediction divided by target, not temporal standard deviation.
- Mouth vertex MAE averages absolute xyz coordinate differences on the separately licensed lip mask.
- Velocity and acceleration use actual time deltas and never span different trajectory blocks.

Jaw correlation uses the first axis-angle coordinate. Jaw MAE averages all three coordinates.
Descriptor errors are normalized units; jaw is radians; vertex MAE retains native FLAME units.
The geometry CLI can evaluate user-supplied aligned `[T,V,3]` arrays with a user-supplied mask, or
the bundled scalar excerpts. It does not distribute the FLAME topology, mask or full vertex arrays.

## Robot-SyncNet3D and Teacher Boundaries

The Speech-v2 scratch experiment resamples timestamp-driven Simulator motion to 25 Hz and uses
five samples in a 200 ms interval. Eval uses MFCC/L2; Loss uses Mel/cosine+ranking.
Later ordered-ranking experiments use a 30-Hz six-sample backbone and aggregate 600 ms contexts.
In these **3D** offset curves audio is fixed and motion is queried at `t+offset`; its sign convention
is distinct from the released historical 2D wrapper. Both signs and temporal supports are documented,
not conflated into a shared frame-index offset.

The 2D teacher's amplitude ratios use temporal standard deviation relative to original A2F,
unlike Simulator P98-P2 ratios against Hybrid targets. Direct per-clip optimization is not shared
Refiner learning. Small supervised feature gains do not prove better physical articulation.
The renderer's mean-texture mouth artifact and objective hacking remain known limitations.
