#!/usr/bin/env python3
"""Compose already-aligned local videos left-to-right; muted review only."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.visualization.video import compose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=float, default=25)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    compose(args.inputs, args.output, height=args.height, fps=args.fps, ffmpeg=args.ffmpeg)


if __name__ == "__main__":
    main()
