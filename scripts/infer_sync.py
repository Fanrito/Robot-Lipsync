#!/usr/bin/env python3
"""Optional inference with independently obtained original SyncNet assets."""
import argparse
import csv
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.syncnet.sequence import evaluate_sample
from src.visualization.video import probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--syncnet-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True, help="Prepared BGR 224x224 video at 25 Hz")
    parser.add_argument("--audio", type=Path, required=True, help="16-kHz mono int16 PCM WAV")
    parser.add_argument("--spatial-protocol", choices=("full_frame", "upstream_crop", "custom_crop"), required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--offsets-ms", nargs="+", type=int, default=[-400, -200, -100, 0, 100, 200, 400])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or len(set(args.offsets_ms)) != len(args.offsets_ms) or 0 not in args.offsets_ms or any(x % 10 for x in args.offsets_ms):
        raise ValueError("Positive batch size; unique offsets on the legacy 10ms MFCC grid, including zero")
    video = next(s for s in probe(args.video)["streams"] if s["codec_type"] == "video")
    num, den = map(float, video["avg_frame_rate"].split("/"))
    if video["width"] != 224 or video["height"] != 224 or abs(num/den-25) > 0.01:
        raise ValueError("Prepare 25-Hz 224x224 input explicitly; do not silently change spatial protocol")
    sr, audio = wavfile.read(args.audio)
    if sr != 16000 or audio.ndim != 1 or audio.dtype != np.int16:
        raise ValueError("Expected 16-kHz mono signed 16-bit PCM")
    spec = importlib.util.spec_from_file_location("upstream_syncnet", args.syncnet_dir / "SyncNetModel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.S()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("model_state_dict", payload)
    # Older upstream checkpoints omit only BatchNorm's integer counters.
    for key, value in model.state_dict().items():
        if key.endswith("num_batches_tracked") and key not in state:
            state[key] = value
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    torch.set_num_threads(2)
    curve, summary = evaluate_sample(model, args.video, args.audio, args.offsets_ms, device, args.batch_size)
    median = float(np.median([r["mean_distance"] for r in curve]))
    rows = [dict(sample_id=args.sample_id, category=args.category, spatial_protocol=args.spatial_protocol,
                 **r, relative_score=median-r["mean_distance"], label=int(r["offset_ms"] == 0)) for r in curve]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(summary)


if __name__ == "__main__":
    main()
