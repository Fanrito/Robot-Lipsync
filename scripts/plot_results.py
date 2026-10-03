#!/usr/bin/env python3
"""Regenerate every public plot using only bundled numeric evidence."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--output", type=Path, default=Path("assets/plots"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f8fafb", "axes.facecolor": "#f8fafb"})
    blue, orange, green = "#156b8a", "#c26b31", "#34846a"

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(args.output / name, dpi=160, bbox_inches="tight")
        plt.close(fig)

    rows = read(args.results / "sync2d_test_curves.csv")
    groups = defaultdict(list)
    for r in rows:
        groups[r["sample_id"]].append(r)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    values = []
    for group in groups.values():
        group = sorted(group, key=lambda r: float(r["offset_ms"]))
        x = [float(r["offset_ms"]) for r in group]
        y = [float(r["relative_score"]) for r in group]
        values.append(y)
        ax.plot(x, y, color=blue, alpha=.14, lw=1)
    ax.plot(x, np.mean(values, axis=0), "o-", color=blue, lw=2.3, label="Mean over 8 held-out utterances")
    ax.axvline(0, color="grey", lw=.8, ls="--")
    ax.set(title="2D SyncNet: full-frame, full-sequence offset search", xlabel="Audio query shift (ms)", ylabel="Median distance - distance (higher is better)")
    ax.legend(frameon=False)
    save(fig, "sync_offset_curve.png")

    labels = [int(r["label"]) for r in rows]
    scores = [float(r["relative_score"]) for r in rows]
    fpr, tpr, _ = roc_curve(labels, scores)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(fpr, tpr, color=blue, label=f"AUC = {roc_auc_score(labels, scores):.4f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=.8)
    axes[0].set(title="56 sequence-offset pairs / 8 clips", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[0].legend(frameon=False)
    offsets = [float(min(group, key=lambda r: float(r["mean_distance"]))["offset_ms"]) for group in groups.values()]
    unique, counts = np.unique(offsets, return_counts=True)
    axes[1].bar(unique, counts, width=50, color=blue)
    axes[1].set(title=f"Offset localization: MAE {np.mean(np.abs(offsets)):.0f} ms", xlabel="Predicted offset (ms)", ylabel="Utterance count", xticks=[-100, 0, 100], yticks=range(0, 9, 2))
    save(fig, "sync_roc_and_offsets.png")

    rows = read(args.results / "robot_flame_metrics.csv")
    c = [r for r in rows if r["domain"] == "group_b_v2_controlled"]
    s = [r for r in rows if r["domain"] == "speech_v2_test"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    axes[0].bar(range(len(c)), [float(r["mouth_vertex_mae"]) for r in c], color=["#bac5c8"]*3+[blue, orange])
    axes[0].set_xticks(range(len(c)), ["Static", "+ static\nresidual", "Temporal\nscratch", "+ temporal\n(history=12)", "Current\nonly"])
    axes[0].set(title="Controlled test: coordinate-wise mouth MAE", ylabel="FLAME coordinate units (lower is better)")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    keys = ["inner_lip_open_correlation", "outer_lip_open_correlation", "mouth_width_correlation", "jaw_correlation", "asymmetry_correlation"]
    x = np.arange(len(keys))
    for r, shift, color, label in zip(s, [-.19, .19], [orange, blue], ["Frozen V1", "Speech-v2 adapted"]):
        axes[1].bar(x+shift, [float(r[k]) for k in keys], .38, color=color, label=label)
    axes[1].set_xticks(x, ["Inner", "Outer", "Width", "Jaw", "Asymmetry"])
    axes[1].set(title="Speech v2: 31 held-out utterances", ylabel="Correlation with Hybrid weak target", ylim=(-.3, 1))
    axes[1].legend(frameon=False, fontsize=9)
    save(fig, "simulator_summary.png")

    rows = read(args.results / "speech_v2_descriptor_excerpt.csv")
    samples = sorted({r["sample_id"] for r in rows})
    fig, axes = plt.subplots(len(samples), 2, figsize=(10, 5.3), squeeze=False)
    for i, sample in enumerate(samples):
        group = [r for r in rows if r["sample_id"] == sample]
        t = [float(r["time_sec"]) for r in group]
        for j, key in enumerate(["outer", "jaw"]):
            for suffix, label, color in [("target", "Hybrid target (weak)", orange), ("pred", "Simulator prediction", blue)]:
                axes[i, j].plot(t, [float(r[f"{key}_{suffix}"]) for r in group], label=label, color=color, lw=1.5)
            axes[i, j].set(title=f"{sample} / {key}", xlabel="Recorded host time relative to clip (s)", ylabel="Normalized opening" if key == "outer" else "Jaw axis-angle x (rad)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, "speech_motion_curves.png")

    ordered = json.loads((args.results / "evidence/sync3d_ordered_ranking.json").read_text())["test"]["offset_curve"]
    x = sorted(map(int, ordered))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(x, [ordered[str(k)]["utterance_mean"] for k in x], "o-", color=green)
    ax.set(title="3D ordered-ranking guard: mean curve is not reliable per-clip alignment", xlabel="Motion query shift (ms)", ylabel="Mean score over 31 test utterances")
    ax.text(.03, .07, "0 ms Top-1 = 6/31; AUC = 0.6598\nA coarse guard, not a solved fine-sync objective", transform=ax.transAxes, fontsize=9)
    save(fig, "sync3d_guard_curve.png")

    rows = read(args.results / "teacher_direct_optimization.csv")
    wanted = ["A0_absolute_cap08_weak", "B2_centered_cap04", "D_centered_shape_amp_guard"]
    rows = [next(r for r in rows if r["config"] == n) for n in wanted]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(3)
    for key, shift, label, color in [("inner", -.24, "Inner opening", blue), ("outer", 0, "Outer opening", orange), ("jaw", .24, "Jaw", green)]:
        ax.bar(x+shift, [float(r[f"{key}_amplitude_ratio_mean"]) for r in rows], .24, label=label, color=color)
    ax.axhline(1, color="grey", ls="--", lw=1)
    ax.set_xticks(x, ["Absolute feature\nweak constraints", "Centered feature\nno geometry guard", "Centered + shape\n+ amplitude + sync guard"])
    ax.set(title="2D teacher direct optimization: amplitude shortcut and constraints", ylabel="Temporal amplitude / original A2F (8 short clips)")
    ax.legend(frameon=False)
    save(fig, "teacher_constraints.png")


if __name__ == "__main__":
    main()
