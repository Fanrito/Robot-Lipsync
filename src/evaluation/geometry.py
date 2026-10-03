"""Original Robot-FLAME descriptor and temporal metrics, unchanged."""
import numpy as np
import torch

def safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) < 3 or np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return None
    return float(np.corrcoef(left, right)[0, 1])

def robust_span(values: np.ndarray) -> float:
    return float(np.percentile(values, 98) - np.percentile(values, 2))

def normalize_landmarks(landmarks: torch.Tensor) -> torch.Tensor:
    center = 0.5 * (landmarks[:, 0, :2] + landmarks[:, 16, :2])
    scale = torch.linalg.vector_norm(landmarks[:, 16, :2] - landmarks[:, 0, :2], dim=-1)
    return (landmarks[:, :, :2] - center[:, None]) / scale[:, None, None].clamp_min(1e-5)

def mouth_descriptors(landmarks: torch.Tensor) -> torch.Tensor:
    return torch.stack((
        torch.linalg.vector_norm(landmarks[:, 62] - landmarks[:, 66], dim=-1),
        torch.linalg.vector_norm(landmarks[:, 51] - landmarks[:, 57], dim=-1),
        torch.linalg.vector_norm(landmarks[:, 48] - landmarks[:, 54], dim=-1),
        torch.abs(landmarks[:, 48, 1] - landmarks[:, 54, 1]),
    ), dim=-1)

def derivative_errors(values: np.ndarray, target: np.ndarray, times_ns: np.ndarray, blocks: np.ndarray) -> dict:
    velocity_errors, acceleration_errors = [], []
    for block in np.unique(blocks):
        selected = np.flatnonzero(blocks == block)
        if len(selected) < 3:
            continue
        dt = np.diff(times_ns[selected]).astype(np.float64) / 1e9
        pred_velocity = np.diff(values[selected], axis=0) / dt[:, None]
        target_velocity = np.diff(target[selected], axis=0) / dt[:, None]
        velocity_errors.append((pred_velocity - target_velocity) ** 2)
        acc_dt = 0.5 * (dt[1:] + dt[:-1])
        pred_acc = np.diff(pred_velocity, axis=0) / acc_dt[:, None]
        target_acc = np.diff(target_velocity, axis=0) / acc_dt[:, None]
        acceleration_errors.append((pred_acc - target_acc) ** 2)
    return {
        "velocity_rmse": float(np.sqrt(np.mean(np.concatenate(velocity_errors)))) if velocity_errors else None,
        "acceleration_rmse": float(np.sqrt(np.mean(np.concatenate(acceleration_errors)))) if acceleration_errors else None,
    }
