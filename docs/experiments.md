# Research Record

## 1. Establish What 2D Synchronization Scores Mean

Original SyncNet weights can discriminate large offsets on some robot inputs, but input crop
changes the outcome. Full-sequence statistics recover information lost by treating independent
short windows as final decisions. A synthetic Wav2Lip set also exhibited a systematic offset,
so treating all generated clips as exactly synchronized positives was unsafe.
Protocol-controlled v2 training improved offset localization but not AUC on the small real test set.

## 2. Learn Robot Motion, Not Human Motion Amplitudes

The public human-domain mapper alone over-predicts controlled outer-opening amplitude
(approximately 2.91x). A temporal residual learns command history and within-command response.
Controlled-domain static+temporal outperforms static-only, static-residual and current-only ablations.
Speech combinations introduce another domain gap, motivating a robot static adapter and later
train-only speech adaptation. Speech v2 preserves measured command and frame timestamps.

The Speech-v2 adaptation checkpoint was selected on validation loss after eight low-learning-rate
epochs, never test optimization. It improves outer-opening correlation to 0.841 and amplitude to
0.966, but inner opening and asymmetry remain weak. Reconstruction quality still limits targets;
there is no independent 3D scanner ground truth.

## 3. A Good Mean Sync Curve Is Not a Safe Optimization Loss

LatentSync-generated Hybrid-FLAME positives initially led to a motion-domain mismatch with the
Simulator. Noisy real-video reconstructions were also a poor direct motion-positive source.
The current direction evaluates audio and Simulator(actual recorded commands) on the same timeline.
Human-pretrained transfers, audio-prior/positional-encoding ablations and ordered-ranking experiments
were completed privately. Selected final statistics are released, but internal model implementations
and checkpoints are not.

The ordered-ranking mean curve has a zero-offset peak, while per-utterance Top-1 is only 6/31.
This rules out claiming a solved fine temporal objective. A restricted sync guard plus independent
geometry/dynamics checks is more defensible than maximizing the score alone.

## 4. Treat a 2D Teacher as a Constrained Prior

LatentSync remains in 2D. Rendered Robot-FLAME and teacher video are compared through frozen
lipreading features and normalized FAN observations. Unconstrained feature reduction can inflate
mouth opening or exploit texture. Direct optimization with geometry, amplitude and dynamics
constraints reduces those shortcuts, but it is not evidence of unseen-utterance generalization.

A 40/8/8 shared-Refiner ablation yields small, bounded residuals. Feature loss improves in 7/8
held-out clips; FAN shape improves in 3/8 and bilabial closure is inconsistent. The no-teacher
objective has an exact identity solution. This is a useful negative control, not a training failure.
No large-scale Refiner benefit or robot execution improvement is claimed.

## 5. Physical Execution Still Needs Measurement

Subsequent repeated A/B execution audits distinguish requested blendshape changes, quantized motor
targets, and observed mouth motion. Tiny learned residuals do not guarantee measurable physical
movement. Landmark resolution and run-to-run variability constrain what can be concluded.
Hardware configurations, control scripts, microphone recordings and execution logs are withheld.

## Priorities

1. Independent, higher-resolution mouth observations and repeated held-out execution tests.
2. Fine-offset losses that retain a reliable per-utterance basin without amplitude/jitter shortcuts.
3. Rights-cleared data/weight subsets and a fully reproducible reconstruction/training package.
