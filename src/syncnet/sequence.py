"""Project full-sequence wrapper; provide an independently licensed SyncNet model.

Input is preprocessed BGR uint8 video at 25 Hz and 16-kHz mono PCM audio.
Positive offset compares video(t) with audio(t + offset). Not the 3D protocol.
Original wrapper body is retained, including per-offset valid overlap lengths.
"""
from pathlib import Path
import cv2
import numpy as np
import python_speech_features
import torch
from scipy.io import wavfile

def batched_forward_video(model, frames: np.ndarray, device: torch.device, batch_size: int) -> torch.Tensor:
    windows = []
    for start in range(max(0, len(frames) - 4)):
        windows.append(np.transpose(frames[start:start + 5], (3, 0, 1, 2)))
    outputs = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            tensor = torch.from_numpy(np.stack(windows[start:start + batch_size]).astype(np.float32)).to(device)
            outputs.append(model.forward_lip(tensor).cpu())
    return torch.cat(outputs) if outputs else torch.empty((0, 1024))

def batched_forward_audio(model, mfcc: np.ndarray, starts: list[int], device: torch.device, batch_size: int) -> dict[int, torch.Tensor]:
    unique = sorted(set(starts))
    outputs: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for index in range(0, len(unique), batch_size):
            current = unique[index:index + batch_size]
            windows = np.stack([mfcc[:, start:start + 20][None] for start in current]).astype(np.float32)
            features = model.forward_aud(torch.from_numpy(windows).to(device)).cpu()
            outputs.update({start: feature for start, feature in zip(current, features)})
    return outputs

def evaluate_sample(model, video: Path, audio: Path, offsets_ms: list[int], device: torch.device, batch_size: int) -> tuple[list[dict[str, float]], dict[str, float]]:
    cap = cv2.VideoCapture(str(video)); frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) < 6:
        raise RuntimeError("视频帧不足")
    video_features = batched_forward_video(model, np.stack(frames), device, batch_size)
    sample_rate, waveform = wavfile.read(audio)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    mfcc = np.asarray(python_speech_features.mfcc(waveform, sample_rate), dtype=np.float32).T
    shifts = {offset: round(offset / 10) for offset in offsets_ms}
    valid_by_offset: dict[int, list[tuple[int, int]]] = {}
    all_audio_starts = []
    for offset, shift in shifts.items():
        pairs = []
        for video_index in range(len(video_features)):
            audio_start = video_index * 4 + shift
            if 0 <= audio_start and audio_start + 20 <= mfcc.shape[1]:
                pairs.append((video_index, audio_start)); all_audio_starts.append(audio_start)
        if not pairs:
            raise RuntimeError(f"offset {offset} ms 没有有效重叠窗口")
        valid_by_offset[offset] = pairs
    audio_features = batched_forward_audio(model, mfcc, all_audio_starts, device, batch_size)
    curve = []
    for offset in offsets_ms:
        pairs = valid_by_offset[offset]
        v = torch.stack([video_features[index] for index, _ in pairs])
        a = torch.stack([audio_features[start] for _, start in pairs])
        distances = torch.nn.functional.pairwise_distance(v, a).numpy()
        curve.append({"offset_ms": offset, "mean_distance": float(np.mean(distances)), "median_distance": float(np.median(distances)), "std_distance": float(np.std(distances)), "window_count": len(distances)})
    best = min(curve, key=lambda row: row["mean_distance"])
    values = np.asarray([row["mean_distance"] for row in curve])
    summary = {"predicted_offset_ms": int(best["offset_ms"]), "offset_abs_error_ms": abs(int(best["offset_ms"])), "confidence": float(np.median(values) - np.min(values))}
    return curve, summary
