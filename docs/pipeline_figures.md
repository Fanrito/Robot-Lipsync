# Pipeline Figures

Two project-owned schematics accompany the research showcase. They depict the implemented
experimental protocols, not a claim that every component is shipped or every research question
is solved. No model weights, robot SDK, licensed geometry, private recordings or third-party
artwork is used to draw them.

| Figure | Preview | Editable artwork | Vector export |
|---|---|---|---|
| System overview | [PNG](../assets/figures/robot_lipsync_pipeline.png) | [SVG](../assets/figures/robot_lipsync_pipeline.svg) | [PDF](../assets/figures/robot_lipsync_pipeline.pdf) |
| Simulator architecture | [PNG](../assets/figures/robot_flame_simulator.png) | [SVG](../assets/figures/robot_flame_simulator.svg) | [PDF](../assets/figures/robot_flame_simulator.pdf) |

## Reading the Overview

- **(a) Physical execution.** Audio2Face emits 51D blendshapes at 30 Hz. The logged 3-to-1
  selection identifies actual 10 Hz commands before the unchanged vendor mapping. Playback,
  A2F source frames, commands and camera frames share a traceable host-monotonic timeline.
  The camera rate is measured, not inferred from a nominal 30 fps.
- **(b) Differentiable surrogate.** Twelve actual past/current commands and measured interval
  phase predict FLAME expression and jaw. The name-aligned frozen static mapper is calibrated
  by a robot adapter, with a TCN residual for dynamics. Shared MICA identity and canonical pose
  produce 5023-vertex geometry. MICA + EMOCA + DECA reconstruction supplies **weak Simulator
  supervision**, not independently measured 3D ground truth.
- **(c) Distinct evaluation/objective branches.** 2D SyncNet uses audio and real video with
  full-sequence offset search. Current 3D positives use aligned audio and Simulator-produced
  mouth motion. LatentSync stays in **2D**, compared to differentiable renders in lipreading
  feature space. It is not the current SyncNet3D positive-motion source. Bounded residual
  optimization and Refiner feasibility remain exploratory; physical improvements are unproven.

Solid arrows indicate data/inference; the dashed arrow indicates weak supervision. Color is
redundant with labels: blue for acquisition/2D sync, teal for the differentiable Simulator/3D
sync, ochre for weak supervision and exploratory teacher objectives. The rows are staged
subfigures: repeated labels identify the same recorded command, video and audio streams,
without drawing crossing wires between every subfigure.

## Reading the Architecture Detail

The source of truth is [model.py](../src/robot_flame/model.py), especially
`NoetixBlendshapeAdapter`, `FrozenMediaPipeToFLAME`, `RobotStaticAdapter`, `TemporalEncoder`
and `BlendshapeRobotFLAME.forward`.

The selected external mapper expects **MediaPipe neutral + 51 shared channels**, not arbitrary
ARKit52 concatenation. Its pinned matrices are `[52,50]` and `[52,6]`; the last three pose
coordinates supply jaw. The MLP adapter also consumes the current normalized command;
the affine adapter uses scale and bias only. Normalization is trained on the training split.
The TCN has hidden width 192, dilations 1/2/4/8, explicit phase features and bounded residual
heads. It uses symmetric padding over a finite past-command window, not a redesigned streaming
causal convolution. The schematic history bars are illustrative, **not experimental data**.

Frozen/static and learned colors describe Simulator training. The adapted Simulator, including
its learned components, is frozen for the downstream gradient/Refiner experiments.
See [methodology](methodology.md) for temporal window protocols and known limitations.

## Regenerate or Edit

From the repository root, using the existing public requirements:

```bash
python scripts/draw_pipeline.py
python scripts/draw_pipeline.py --figure overview --dpi 240 --output outputs/figures
python -m unittest discover -s tests -v
```

[draw_pipeline.py](../scripts/draw_pipeline.py) uses Matplotlib's non-interactive CPU backend;
no browser, Graphviz, LaTeX installation, GPU, network or robot is required. It emits both
figures as 2880-pixel-wide PNGs at the default 180 dpi, plus SVG and PDF. SVG retains editable
text; PDF embeds TrueType fonts. DejaVu fonts ship with Matplotlib. Export timestamps are
removed and SVG IDs are seeded for repeatable output in the same dependency environment.

Change layout/text in the script to keep all formats consistent. Export-time checks reject
text that leaves its canvas or an explicitly bounded card. These checks do not replace visual
inspection for overlapping arrows or scientific accuracy.
