# Attribution and Third-Party Boundaries

The released Python source is selected original project integration/model/evaluation work.
It does not include upstream network definitions, SDK mapping code or trained weights.
No rights to upstream assets are granted by this repository's MIT license.

| Upstream | Use in the private research pipeline | Release treatment |
|---|---|---|
| [SyncNet](https://github.com/joonson/syncnet_python) | Original 2D audio/visual embeddings | Model-injected wrapper only; independently obtain code/weights |
| [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) | Earlier synthetic talking-video/domain-adaptation experiments | No implementation, weights or generated examples bundled |
| [LatentSync](https://github.com/bytedance/LatentSync) | 2D talking-face teacher | Attribution and aggregate experiments only |
| [FLAME2020](https://flame.is.tue.mpg.de/modellicense.html) | Parametric head geometry | No model, mesh arrays, blend bases or mask bundled |
| [DECA](https://github.com/yfeng95/DECA) | Jaw/pose and reconstruction utilities | No code/weights/texture bundled |
| [MICA](https://github.com/Zielon/MICA) | Fixed robot identity | No code, identity coefficients or weights bundled |
| [EMOCA](https://github.com/radekd91/emoca) | Expression weak targets | No code or weights bundled |
| [MediaPipe-to-FLAME](https://github.com/PeizhiYan/mediapipe-blendshapes-to-flame/tree/34c4964cdb9516540900a556dba1a71493f07138) | Frozen 52-to-50D expression / pose matrix mapping | Project-written name adapter only; mapping matrices excluded |
| [Learn2Talk](https://arxiv.org/abs/2404.12888) | 3D synchronization and lipreading-supervision research reference | No internal implementation, dataset, mask or checkpoint bundled |
| [FAN / face-alignment](https://github.com/1adrianb/face-alignment) | 2D observations and crop protocol | No detector code or weights bundled |
| PyTorch, NumPy, scikit-learn, Matplotlib, FFmpeg | Numerical, learning and media tools | External dependencies under their respective licenses |

FLAME2020, MICA and DECA assets have research-use restrictions; the newer FLAME Open license
must not be assumed to apply to FLAME2020. The selected mapper revision did not provide a
standard standalone license and documented research/education restrictions. These materials
are deliberately omitted rather than relicensed. Consult each upstream for current terms.

The three MP4s and previews are **rendered non-commercial research-result illustrations** on the
robot, not a distribution of model/texture assets. Media and result artifacts are excluded from
the source-code MIT grant; no downstream commercial reuse or sublicensing permission is asserted.
Third-party textured teacher renders, human-subject clips, raw camera recordings, native microphone
audio and synthesized voices are not released. For uses beyond viewing this research record,
obtain any required upstream and robot-image permissions separately.

FLAME acknowledgement: Tianye Li, Timo Bolkart, Michael J. Black, Hao Li and Javier Romero,
*Learning a Model of Facial Shape and Expression from 4D Scans*, ACM TOG / SIGGRAPH Asia, 2017.
[Paper](https://doi.org/10.1145/3130800.3130813).
SyncNet acknowledgement: Joon Son Chung and Andrew Zisserman,
*Out of Time: Automated Lip Sync in the Wild*, ACCV workshop, 2016.

These are upstream acknowledgements, not publications claimed for Robot-Lipsync.
