"""Blendshape-conditioned differentiable Robot-FLAME simulators."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


NOETIX_51_NAMES = (
    "EyeBlinkLeft", "EyeBlinkRight", "EyeSquintLeft", "EyeSquintRight",
    "EyeLookDownLeft", "EyeLookDownRight", "EyeLookInLeft", "EyeLookInRight",
    "EyeWideLeft", "EyeWideRight", "EyeLookOutLeft", "EyeLookOutRight",
    "EyeLookUpLeft", "EyeLookUpRight", "BrowDownLeft", "BrowDownRight",
    "BrowInnerUp", "BrowOuterUpLeft", "BrowOuterUpRight", "JawOpen",
    "MouthClose", "JawLeft", "JawRight", "JawForward", "MouthUpperUpLeft",
    "MouthUpperUpRight", "MouthLowerDownLeft", "MouthLowerDownRight",
    "MouthRollUpper", "MouthRollLower", "MouthSmileLeft", "MouthSmileRight",
    "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft",
    "MouthStretchRight", "MouthFrownLeft", "MouthFrownRight", "MouthPressLeft",
    "MouthPressRight", "MouthPucker", "MouthFunnel", "MouthLeft", "MouthRight",
    "MouthShrugLower", "MouthShrugUpper", "NoseSneerLeft", "NoseSneerRight",
    "CheekPuff", "CheekSquintLeft", "CheekSquintRight",
)

# MediaPipe's mapper input has a learned neutral category followed by the 51
# ARKit-compatible categories. It does not contain TongueOut.
MEDIAPIPE_52_NAMES = (
    "_neutral", "BrowDownLeft", "BrowDownRight", "BrowInnerUp",
    "BrowOuterUpLeft", "BrowOuterUpRight", "CheekPuff", "CheekSquintLeft",
    "CheekSquintRight", "EyeBlinkLeft", "EyeBlinkRight", "EyeLookDownLeft",
    "EyeLookDownRight", "EyeLookInLeft", "EyeLookInRight", "EyeLookOutLeft",
    "EyeLookOutRight", "EyeLookUpLeft", "EyeLookUpRight", "EyeSquintLeft",
    "EyeSquintRight", "EyeWideLeft", "EyeWideRight", "JawForward", "JawLeft",
    "JawOpen", "JawRight", "MouthClose", "MouthDimpleLeft", "MouthDimpleRight",
    "MouthFrownLeft", "MouthFrownRight", "MouthFunnel", "MouthLeft",
    "MouthLowerDownLeft", "MouthLowerDownRight", "MouthPressLeft",
    "MouthPressRight", "MouthPucker", "MouthRight", "MouthRollLower",
    "MouthRollUpper", "MouthShrugLower", "MouthShrugUpper", "MouthSmileLeft",
    "MouthSmileRight", "MouthStretchLeft", "MouthStretchRight",
    "MouthUpperUpLeft", "MouthUpperUpRight", "NoseSneerLeft", "NoseSneerRight",
)

ARKIT_52_NAMES = NOETIX_51_NAMES + ("TongueOut",)


def validate_noetix_schema(names: tuple[str, ...] | list[str]) -> None:
    names = tuple(names)
    if names != NOETIX_51_NAMES:
        missing = sorted(set(NOETIX_51_NAMES) - set(names))
        extra = sorted(set(names) - set(NOETIX_51_NAMES))
        raise ValueError(
            "Noetix blendshape schema/order mismatch; "
            f"missing={missing}, extra={extra}, exact_order={names == NOETIX_51_NAMES}"
        )


class NoetixBlendshapeAdapter(nn.Module):
    """Name-based Noetix51 conversion to ARKit52 and MediaPipe52 schemas."""

    def __init__(self, neutral_mode: str = "complement_max") -> None:
        super().__init__()
        validate_noetix_schema(NOETIX_51_NAMES)
        if neutral_mode not in {"complement_max", "zero"}:
            raise ValueError(f"unsupported neutral_mode: {neutral_mode}")
        self.neutral_mode = neutral_mode
        source = {name: index for index, name in enumerate(ARKIT_52_NAMES)}
        shared = MEDIAPIPE_52_NAMES[1:]
        if set(shared) != set(NOETIX_51_NAMES):
            raise RuntimeError("MediaPipe and Noetix shared channel sets differ")
        self.register_buffer(
            "mediapipe_gather", torch.tensor([source[name] for name in shared], dtype=torch.long)
        )

    def to_arkit52(self, blendshape_51d: torch.Tensor) -> torch.Tensor:
        if blendshape_51d.shape[-1] != 51:
            raise ValueError(f"expected [...,51], got {tuple(blendshape_51d.shape)}")
        tongue = torch.zeros_like(blendshape_51d[..., :1])
        return torch.cat((blendshape_51d, tongue), dim=-1)

    def to_mediapipe52(self, blendshape_51d: torch.Tensor) -> torch.Tensor:
        if blendshape_51d.shape[-1] != 51:
            raise ValueError(f"expected [...,51], got {tuple(blendshape_51d.shape)}")
        # Materialize the requested ARKit52 schema first (TongueOut=0), then
        # adapt by name to this mapper's MediaPipe52 schema.
        arkit52 = self.to_arkit52(blendshape_51d)
        shared = arkit52.index_select(-1, self.mediapipe_gather)
        if self.neutral_mode == "complement_max":
            neutral = (1.0 - shared.amax(dim=-1, keepdim=True)).clamp(0.0, 1.0)
        else:
            neutral = torch.zeros_like(shared[..., :1])
        return torch.cat((neutral, shared), dim=-1)


class FrozenMediaPipeToFLAME(nn.Module):
    """Pinned PeizhiYan linear mapper without modifying third-party code."""

    def __init__(self, mappings_dir: Path, neutral_mode: str = "complement_max") -> None:
        super().__init__()
        expression = np.load(mappings_dir / "bs2exp.npy").astype(np.float32)
        pose = np.load(mappings_dir / "bs2pose.npy").astype(np.float32)
        if expression.shape != (52, 50) or pose.shape != (52, 6):
            raise ValueError(
                f"expected pinned 52x50/52x6 mappings, got {expression.shape}/{pose.shape}"
            )
        self.adapter = NoetixBlendshapeAdapter(neutral_mode)
        self.register_buffer("expression_matrix", torch.from_numpy(expression))
        self.register_buffer("pose_matrix", torch.from_numpy(pose))

    def forward(self, blendshape_51d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mediapipe = self.adapter.to_mediapipe52(blendshape_51d)
        expression = mediapipe @ self.expression_matrix
        pose = mediapipe @ self.pose_matrix
        return expression, pose[..., 3:6]


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.conv(values).transpose(1, 2)
        update = self.dropout(F.gelu(self.norm(update))).transpose(1, 2)
        return values + update


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int = 51, hidden_dim: int = 192, dropout: float = 0.1) -> None:
        super().__init__()
        self.input = nn.Conv1d(input_dim, hidden_dim, 1)
        self.blocks = nn.ModuleList(
            ResidualTCNBlock(hidden_dim, dilation, dropout) for dilation in (1, 2, 4, 8)
        )
        self.phase = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 32))
        self.output = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )

    def forward(self, history: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        values = self.input(history.transpose(1, 2))
        for block in self.blocks:
            values = block(values)
        phase_input = torch.stack(
            (phase, torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)), dim=-1
        )
        return self.output(torch.cat((values[..., -1], self.phase(phase_input)), dim=-1))


class StaticResidualEncoder(nn.Module):
    def __init__(self, input_dim: int = 51, hidden_dim: int = 192) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + 3, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )

    def forward(self, history: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        phase_input = torch.stack(
            (phase, torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)), dim=-1
        )
        return self.network(torch.cat((history[:, -1], phase_input), dim=-1))


class RobotStaticAdapter(nn.Module):
    """Calibrate the human-domain static FLAME prior for Noetix commands."""

    def __init__(self, kind: str, hidden_dim: int = 128) -> None:
        super().__init__()
        if kind not in {"affine", "mlp"}:
            raise ValueError(f"unsupported static adapter: {kind}")
        self.kind = kind
        if kind == "affine":
            self.scale = nn.Parameter(torch.ones(53))
            self.bias = nn.Parameter(torch.zeros(53))
            self.network = None
        else:
            self.register_parameter("scale", None)
            self.register_parameter("bias", None)
            self.network = nn.Sequential(
                nn.Linear(53 + 51, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 53),
            )
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    def forward(self, static_parameters: torch.Tensor, current_normalized: torch.Tensor) -> torch.Tensor:
        if self.kind == "affine":
            return static_parameters * self.scale + self.bias
        return static_parameters + self.network(torch.cat((static_parameters, current_normalized), dim=-1))


class BlendshapeRobotFLAME(nn.Module):
    """Comparable parameter predictors sharing one frozen static prior."""

    MODES = {
        "static", "static_residual", "temporal_scratch", "static_temporal",
        "static_adapter_affine", "static_adapter_mlp",
        "adapter_temporal_affine", "adapter_temporal_mlp",
    }

    def __init__(
        self,
        mappings_dir: Path,
        mode: str,
        command_mean: np.ndarray,
        command_std: np.ndarray,
        hidden_dim: int = 192,
        neutral_mode: str = "complement_max",
        expression_delta_scale: float = 5.0,
        jaw_delta_scale: float = 0.6,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unsupported mode {mode}")
        self.mode = mode
        self.static_mapper = FrozenMediaPipeToFLAME(mappings_dir, neutral_mode)
        self.register_buffer("command_mean", torch.from_numpy(command_mean.astype(np.float32)))
        self.register_buffer("command_std", torch.from_numpy(command_std.astype(np.float32)))
        self.expression_delta_scale = expression_delta_scale
        self.jaw_delta_scale = jaw_delta_scale
        self.static_adapter: RobotStaticAdapter | None = None
        self.encoder: nn.Module | None = None
        self.expression_head: nn.Module | None = None
        self.jaw_head: nn.Module | None = None
        if "adapter" in mode:
            adapter_kind = "affine" if mode.endswith("affine") else "mlp"
            self.static_adapter = RobotStaticAdapter(adapter_kind)
        has_temporal = mode in {
            "static_residual", "temporal_scratch", "static_temporal",
            "adapter_temporal_affine", "adapter_temporal_mlp",
        }
        if has_temporal:
            self.encoder = (
                StaticResidualEncoder(51, hidden_dim)
                if mode == "static_residual"
                else TemporalEncoder(51, hidden_dim)
            )
            self.expression_head = nn.Linear(hidden_dim, 50)
            self.jaw_head = nn.Linear(hidden_dim, 3)
            if mode in {
                "static_residual", "static_temporal",
                "adapter_temporal_affine", "adapter_temporal_mlp",
            }:
                nn.init.zeros_(self.expression_head.weight)
                nn.init.zeros_(self.expression_head.bias)
                nn.init.zeros_(self.jaw_head.weight)
                nn.init.zeros_(self.jaw_head.bias)

    def forward(self, history_raw: torch.Tensor, phase: torch.Tensor) -> dict[str, torch.Tensor]:
        current = history_raw[:, -1]
        static_expression, static_jaw = self.static_mapper(current)
        normalized = (history_raw - self.command_mean) / self.command_std
        static_parameters = torch.cat((static_expression, static_jaw), dim=-1)
        adapted_parameters = (
            self.static_adapter(static_parameters, normalized[:, -1])
            if self.static_adapter is not None else static_parameters
        )
        adapted_expression, adapted_jaw = adapted_parameters[..., :50], adapted_parameters[..., 50:]
        adapter_delta_expression = adapted_expression - static_expression
        adapter_delta_jaw = adapted_jaw - static_jaw
        has_temporal = self.encoder is not None
        if not has_temporal:
            expression, jaw = adapted_expression, adapted_jaw
            temporal_delta_expression = torch.zeros_like(expression)
            temporal_delta_jaw = torch.zeros_like(jaw)
        else:
            features = self.encoder(normalized, phase)
            predicted_expression = self.expression_delta_scale * torch.tanh(self.expression_head(features))
            predicted_jaw = self.jaw_delta_scale * torch.tanh(self.jaw_head(features))
            if self.mode in {
                "static_residual", "static_temporal",
                "adapter_temporal_affine", "adapter_temporal_mlp",
            }:
                temporal_delta_expression, temporal_delta_jaw = predicted_expression, predicted_jaw
                expression = adapted_expression + temporal_delta_expression
                jaw = adapted_jaw + temporal_delta_jaw
            else:
                expression, jaw = predicted_expression, predicted_jaw
                temporal_delta_expression = expression - adapted_expression
                temporal_delta_jaw = jaw - adapted_jaw
        delta_expression = expression - static_expression
        delta_jaw = jaw - static_jaw
        return {
            "expression": expression,
            "jaw": jaw,
            "static_expression": static_expression,
            "static_jaw": static_jaw,
            "adapted_expression": adapted_expression,
            "adapted_jaw": adapted_jaw,
            "adapter_delta_expression": adapter_delta_expression,
            "adapter_delta_jaw": adapter_delta_jaw,
            "temporal_delta_expression": temporal_delta_expression,
            "temporal_delta_jaw": temporal_delta_jaw,
            "delta_expression": delta_expression,
            "delta_jaw": delta_jaw,
        }
