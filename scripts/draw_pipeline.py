#!/usr/bin/env python3
"""Draw editable research schematics without models, private data or hardware.

PNG is the README preview; SVG keeps editable text, and PDF is vector artwork.
Only Matplotlib and NumPy from the public requirements are needed.
"""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as PlotPath


ROOT = Path(__file__).resolve().parents[1]
INK = "#182F3B"
MUTED = "#607482"
TEAL = "#067C72"
BLUE = "#316799"
CLAY = "#AB6139"
LINE = "#CFDBDF"
PALE_TEAL = "#EDF7F3"
PALE_BLUE = "#EFF4FA"
PALE_CLAY = "#FCF4EC"
PANEL = "#F7F9FA"


class Diagram:
    """Top-left layout coordinates with export-time text clipping checks."""

    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set(xlim=(0, width), ylim=(height, 0))
        self.ax.axis("off")
        self.text_regions: list[tuple[plt.Text, tuple[float, ...]]] = []

    def text(self, x, y, value, size=12, color=INK, weight="normal",
             ha="left", family="DejaVu Sans", region=None, **kwargs):
        artist = self.ax.text(
            x, y, value.replace("->", r"$\rightarrow$"), fontsize=size, color=color, fontweight=weight,
            ha=ha, va="center", fontfamily=family, linespacing=1.5,
            zorder=4, **kwargs,
        )
        if region is not None:
            self.text_regions.append((artist, region))
        return artist

    def rect(self, x, y, w, h, fill="white", edge=LINE, radius=12,
             linewidth=1, dashed=False):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill, edgecolor=edge, linewidth=linewidth,
            linestyle=(0, (5, 4)) if dashed else "solid", zorder=1,
        )
        self.ax.add_patch(patch)
        return patch

    def arrow(self, points, color=MUTED, dashed=False, width=1.6):
        path = PlotPath(points, [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(points) - 1))
        self.ax.add_patch(FancyArrowPatch(
            path=path, arrowstyle="-|>", mutation_scale=13,
            linewidth=width, color=color,
            linestyle=(0, (4, 4)) if dashed else "solid", zorder=2,
        ))

    def badge(self, x, y, w, text, color, fill):
        self.rect(x, y, w, 28, fill, edge=fill, radius=6)
        self.text(x + w / 2, y + 14, text, size=9.5, color=color,
                  weight="bold", ha="center", region=(x, y, w, 28))

    def card(self, x, y, w, h, title, body, color=BLUE, fill="white",
             title_size=15, body_size=11.2):
        self.rect(x, y, w, h, fill, edge=LINE)
        self.ax.plot((x + 20, x + 49), (y + 19, y + 19), color=color,
                     linewidth=2.8, solid_capstyle="round", zorder=3)
        region = (x + 14, y + 6, w - 28, h - 12)
        self.text(x + 20, y + 43, title, size=title_size, weight="bold", region=region)
        self.text(x + 20, y + 84, body, size=body_size, color=MUTED, region=region)

    def panel(self, y, h, number, title, color, subtitle=None):
        self.rect(45, y, 1510, h, PANEL, edge="#E4EAED", radius=16)
        self.badge(70, y + 21, 44, number, color, "white")
        self.text(130, y + 35, title, size=18, weight="bold")
        if subtitle:
            self.text(75, y + 65, subtitle, size=11.5, color=MUTED)

    def header(self, title, subtitle, tag):
        self.text(60, 29, "ROBOT-LIPSYNC  /  RESEARCH PIPELINE", size=10,
                  color=TEAL, weight="bold")
        self.text(1540, 29, tag, size=9.5, color=MUTED, ha="right")
        self.text(60, 79, title, size=31, weight="bold", family="DejaVu Serif")
        self.text(60, 123, subtitle, size=12.5, color=MUTED)

    def validate_text(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        regions = [(a, (0, 0, self.width, self.height)) for a in self.ax.texts]
        regions.extend(self.text_regions)
        for artist, (x, y, w, h) in regions:
            box = artist.get_window_extent(renderer)
            corners = self.ax.transData.inverted().transform(box.get_points())
            left, top = corners.min(axis=0)
            right, bottom = corners.max(axis=0)
            if left < x - 1 or top < y - 1 or right > x + w + 1 or bottom > y + h + 1:
                raise ValueError(f"Text outside layout region: {artist.get_text()!r}")

    def save(self, output: Path, stem: str, dpi: int):
        self.validate_text()
        output.mkdir(parents=True, exist_ok=True)
        paths = []
        for ext in ("png", "svg", "pdf"):
            path = output / f"{stem}.{ext}"
            metadata = {"Creator": "Robot-Lipsync/scripts/draw_pipeline.py"}
            if ext == "svg":
                metadata["Date"] = None
            elif ext == "pdf":
                metadata.update(CreationDate=None, ModDate=None)
            if ext == "svg":
                buffer = StringIO()
                self.fig.savefig(buffer, format="svg", facecolor="white", metadata=metadata)
                # Matplotlib emits trailing spaces in path data; keep generated diffs clean.
                path.write_text("\n".join(line.rstrip() for line in buffer.getvalue().splitlines()) + "\n",
                                encoding="utf-8")
            else:
                self.fig.savefig(path, dpi=dpi, facecolor="white", metadata=metadata)
            paths.append(path)
        plt.close(self.fig)
        return paths


def overview() -> Diagram:
    d = Diagram(1600, 1210)
    d.header(
        "From audio to a physical robot face",
        "Timestamped execution, a differentiable motion surrogate, and constrained visual supervision.",
        "01 / SYSTEM OVERVIEW",
    )

    d.panel(160, 275, "a", "Timestamped physical execution", BLUE)
    d.text(1525, 195, "REAL ROBOT  /  RECORDED OBSERVATIONS", 9.5, BLUE, ha="right")
    d.card(75, 238, 200, 112, "Speech audio", "TTS / playback\nOriginal waveform")
    d.card(305, 238, 240, 112, "Audio2Face", "51D Noetix / ARKit\n30 Hz source trajectory")
    d.card(575, 238, 260, 112, "Actual commands", "Logged 3-to-1 selection\n51D blendshapes at 10 Hz")
    d.card(865, 238, 300, 112, "Physical robot", "Fixed vendor BS-to-motor\nMechanical facial response")
    d.card(1195, 238, 330, 112, "Camera observation", "Real face video + recorded audio\nActual per-frame timestamps")
    for start, end in ((275, 305), (545, 575), (835, 865), (1165, 1195)):
        d.arrow(((start + 3, 294), (end - 3, 294)), color=BLUE)
    d.rect(75, 373, 1450, 40, "#EAF0F5", edge="#EAF0F5", radius=7)
    d.text(94, 393, "ONE HOST CLOCK", 10.5, BLUE, weight="bold")
    d.text(340, 393, "Audio playback  /  A2F source frames  /  selected commands  /  camera frames", 11.5)

    d.panel(457, 352, "b", "Differentiable Robot-FLAME Simulator", TEAL,
            "Predict robot geometry from actual commands, not from motor inversion.")
    d.text(1525, 492, "SPEECH-v2 ADAPTED  /  FROZEN DOWNSTREAM", 9.5, TEAL, ha="right")
    d.card(75, 565, 240, 125, "Command history", "12 steps x 51 channels\nMeasured interval phase", TEAL)
    d.card(345, 565, 480, 125, "Calibrated prior + dynamics",
           "Frozen static mapper -> robot adapter\n+ bounded temporal TCN residual", TEAL, PALE_TEAL, 16, 12)
    d.card(855, 565, 230, 125, "Expression + jaw", "50D expression\n3D jaw axis-angle", TEAL, title_size=14)
    d.card(1115, 565, 410, 125, "Fixed-identity FLAME2020",
           "Shared MICA identity / canonical pose\n5023-vertex robot-aligned geometry", TEAL, title_size=15, body_size=11.5)
    for start, end in ((315, 345), (825, 855), (1085, 1115)):
        d.arrow(((start + 3, 628), (end - 3, 628)), color=TEAL)
    d.badge(75, 737, 240, "REAL-VIDEO WEAK TARGET", CLAY, PALE_CLAY)
    d.rect(345, 724, 740, 60, PALE_CLAY, edge="#E6D6C9", radius=9)
    d.text(365, 745, "MICA identity + EMOCA expression + DECA jaw/pose", 12, CLAY,
           weight="bold", region=(345, 724, 740, 60))
    d.text(365, 770, "Reconstructed from recorded robot video; used for Simulator supervision.", 10, MUTED,
           region=(345, 724, 740, 60))
    d.arrow(((585, 722), (585, 693)), color=CLAY, dashed=True)
    d.text(1115, 752, "Hybrid reconstruction is a weak prior,\nnot measured 3D ground truth.", 11.5, MUTED)

    d.panel(832, 280, "c", "Evaluation and bounded visual-prior research", INK)
    d.text(1525, 867, "SEPARATE OBJECTIVES, NOT A SINGLE SOLVED SYSTEM", 9, MUTED, ha="right")
    cards = [
        (75, 440, BLUE, PALE_BLUE, "2D AUDIOVISUAL SYNCHRONIZATION",
         "Full-sequence offset search", "Audio + real camera video",
         "Sliding windows -> distance curve -> best offset",
         "Spatial-crop and domain-transfer controls."),
        (545, 450, TEAL, PALE_TEAL, "3D AUDIO-MOTION SYNCHRONIZATION",
         "Robot-SyncNet3D / guard", "Aligned audio + Simulator mouth motion",
         "Neutral-relative VOCA585 / canonical pose",
         "Temporal guard; fine sync remains limited."),
        (1025, 500, CLAY, PALE_CLAY, "2D TEACHER  /  EXPLORATORY",
         "Constrained articulation priors", "LatentSync video -> lipreading features",
         "Rendered Robot-FLAME -> feature comparison",
         "Bounded mouth/jaw residuals; physical gains unproven."),
    ]
    for x, w, color, fill, label, title, inputs, operation, caveat in cards:
        d.rect(x, 914, w, 170, "white", edge=LINE)
        d.rect(x, 914, w, 31, fill, edge=fill, radius=9)
        region = (x + 12, 914, w - 24, 170)
        d.text(x + 18, 930, label, 9.2, color, weight="bold", region=region)
        d.text(x + 18, 971, title, 15, weight="bold", region=region)
        d.text(x + 18, 1005, inputs, 11.2, region=region)
        d.text(x + 18, 1032, operation, 10.4, MUTED, region=region)
        d.text(x + 18, 1064, caveat, 9.8, color, region=region)

    d.arrow(((65, 1145), (105, 1145)), TEAL)
    d.text(117, 1145, "Data / inference", 10.3, MUTED)
    d.arrow(((360, 1145), (400, 1145)), CLAY, dashed=True)
    d.text(412, 1145, "Weak supervision", 10.3, MUTED)
    d.text(1535, 1145, "Architecture overview, not a turnkey training release.", 10.3, MUTED, ha="right")
    d.text(65, 1184, "LatentSync stays in 2D; it is NOT the current SyncNet3D positive-motion source.", 10.5, CLAY)
    return d


def simulator_detail() -> Diagram:
    d = Diagram(1600, 1110)
    d.header(
        "A calibrated prior, with robot-specific dynamics",
        "Blendshape-conditioned Robot-FLAME  /  frozen static mapping + learned adapter + bounded temporal residual.",
        "02 / SIMULATOR DETAIL",
    )

    d.rect(45, 167, 1510, 650, PANEL, edge="#E4EAED", radius=16)
    d.badge(65, 193, 250, "TIMESTAMP-DRIVEN INPUT", BLUE, PALE_BLUE)
    d.rect(65, 295, 250, 310, "white")
    region = (80, 300, 220, 300)
    d.text(85, 330, "Actual 10 Hz", 16, weight="bold", region=region)
    d.text(85, 361, "51D commands", 16, weight="bold", region=region)
    # A deliberately schematic history glyph, not an empirical trajectory.
    for i, h in enumerate((17, 25, 15, 34, 42, 31, 21, 39, 50, 36, 29, 46)):
        d.rect(86 + i * 17, 460 - h, 9, h, "#71B4AB", edge="#71B4AB", radius=2)
    d.text(85, 487, "12 past / current steps", 11, MUTED, region=region)
    d.text(85, 523, "+ measured phase", 12, BLUE, weight="bold", region=region)
    d.text(85, 562, r"$\tau,\ \sin(2\pi\tau),\ \cos(2\pi\tau)$", 12, MUTED, region=region)

    d.badge(400, 193, 420, "STATIC SEMANTICS + AMPLITUDE CALIBRATION", TEAL, PALE_TEAL)
    d.rect(400, 255, 420, 80, "white")
    d.text(420, 282, "Name-based channel conversion", 14, weight="bold",
           region=(400, 255, 420, 80))
    d.text(420, 310, "MediaPipe neutral + 51 shared channels", 11.2, MUTED,
           region=(400, 255, 420, 80))
    d.rect(400, 365, 420, 80, PALE_BLUE, edge="#C9D8E8")
    d.text(420, 393, "Pretrained static mapper", 14, weight="bold",
           region=(400, 365, 420, 80))
    d.text(420, 421, "Frozen 52x50 expression / 52x6 pose matrices", 10.8, BLUE,
           region=(400, 365, 420, 80))
    d.rect(400, 475, 420, 90, PALE_TEAL, edge="#B9DAD0")
    d.text(420, 506, "Robot-specific static adapter", 14, weight="bold",
           region=(400, 475, 420, 90))
    d.text(420, 537, "Affine / MLP calibration of the human prior", 11.2, TEAL,
           region=(400, 475, 420, 90))
    d.arrow(((610, 338), (610, 362)), TEAL)
    d.arrow(((610, 448), (610, 472)), TEAL)
    d.arrow(((315, 335), (355, 335), (355, 295), (397, 295)), BLUE)
    d.text(322, 268, r"$B_t$", 13, BLUE)
    # The MLP adapter also reads the current normalized command.
    d.arrow(((355, 335), (355, 520), (397, 520)), BLUE)

    d.badge(400, 592, 420, "HISTORY + INTERVAL PHASE", TEAL, PALE_TEAL)
    d.rect(400, 645, 420, 135, PALE_TEAL, edge="#B9DAD0")
    d.text(420, 674, "Temporal residual TCN", 15, weight="bold",
           region=(400, 645, 420, 135))
    d.text(420, 709, "Hidden 192 / dilations 1, 2, 4, 8", 11.6, TEAL,
           region=(400, 645, 420, 135))
    d.text(420, 748, "Zero-init heads + scaled tanh bounds", 11.2, MUTED,
           region=(400, 645, 420, 135))
    d.arrow(((315, 565), (355, 565), (355, 712), (397, 712)), BLUE)

    d.ax.add_patch(Circle((960, 520), 27, facecolor="white", edgecolor=TEAL,
                         linewidth=1.8, zorder=3))
    d.text(960, 519, "+", 25, TEAL, ha="center")
    d.arrow(((823, 520), (930, 520)), TEAL)
    d.arrow(((823, 712), (960, 712), (960, 550)), TEAL)
    d.text(845, 487, "Calibrated static", 10.1, TEAL)
    d.text(845, 745, "Temporal correction", 10.1, TEAL)
    d.arrow(((990, 520), (1052, 520)), TEAL)

    d.text(1060, 278, "WHY TWO BRANCHES?", 10, TEAL, weight="bold")
    d.text(1060, 316, "Static: amplitude and channel semantics.\nTemporal: history-dependent robot response.",
           11.6, MUTED)

    d.rect(1055, 475, 200, 95, "white")
    d.text(1155, 504, "FLAME state", 14, weight="bold", ha="center",
           region=(1055, 475, 200, 95))
    d.text(1155, 539, "50D expr + 3D jaw", 11.2, MUTED, ha="center",
           region=(1055, 475, 200, 95))
    d.arrow(((1258, 520), (1287, 520)), TEAL)
    d.rect(1290, 475, 245, 95, PALE_BLUE, edge="#C9D8E8")
    d.text(1412, 504, "FLAME2020", 16, weight="bold", ha="center",
           region=(1290, 475, 245, 95))
    d.text(1412, 539, "5023-vertex mesh", 11.5, BLUE, ha="center",
           region=(1290, 475, 245, 95))
    d.text(1290, 427, "FIXED ROBOT IDENTITY", 10, BLUE, weight="bold")
    d.text(1290, 606, "Shared MICA identity\nCanonical global pose", 11.3, MUTED)
    d.text(1060, 684, "Geometry remains differentiable\nwith respect to blendshape inputs.", 12.3, TEAL)
    d.text(1060, 748, "No vendor mapping changes.\nNo motor-to-blendshape inversion.", 11, MUTED)

    d.rect(65, 842, 1470, 95, "#EAF5F1", edge="#D4E7DF", radius=12)
    d.text(90, 890,
           r"$\hat{y}_t = A_{\mathrm{robot}}\!\left(f_{\mathrm{static}}(B_t), B_t\right)"
           r"+g_{\mathrm{temporal}}\!\left(B_{t-11:t},\tau_t\right)$",
           23, TEAL, region=(65, 842, 1470, 95))

    d.badge(65, 969, 110, "FROZEN", BLUE, PALE_BLUE)
    d.text(193, 983, "Pretrained static matrices + FLAME decoder / fixed identity", 11.6, MUTED)
    d.badge(65, 1011, 110, "LEARNED", TEAL, PALE_TEAL)
    d.text(193, 1025, "Robot adapter + temporal residual; the adapted Simulator is frozen downstream.", 11.6, MUTED)
    d.text(65, 1082, "Implementation: src/robot_flame/model.py  |  Licensed assets / pretrained weights are external.",
           10.3, MUTED)
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/figures")
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution; vector exports stay vector")
    parser.add_argument("--figure", choices=("all", "overview", "simulator"), default="all")
    args = parser.parse_args()
    if not 72 <= args.dpi <= 600:
        parser.error("--dpi must be between 72 and 600")
    with matplotlib.rc_context({
        "font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "robot-lipsync-pipeline-v1",
        "pdf.fonttype": 42, "text.usetex": False,
    }):
        figures = {"overview": (overview, "robot_lipsync_pipeline"),
                   "simulator": (simulator_detail, "robot_flame_simulator")}
        for key, (build, stem) in figures.items():
            if args.figure in ("all", key):
                for path in build().save(args.output, stem, args.dpi):
                    print(path)


if __name__ == "__main__":
    main()
