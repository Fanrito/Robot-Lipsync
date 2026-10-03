#!/usr/bin/env python3
"""Architecture-only CPU/gradient test with artificial mapper matrices, not inference."""
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.robot_flame.model import BlendshapeRobotFLAME, NOETIX_51_NAMES, validate_noetix_schema


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    validate_noetix_schema(NOETIX_51_NAMES)
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        np.save(path / "bs2exp.npy", rng.normal(0, .01, (52, 50)).astype(np.float32))
        np.save(path / "bs2pose.npy", rng.normal(0, .01, (52, 6)).astype(np.float32))
        model = BlendshapeRobotFLAME(path, "adapter_temporal_mlp", np.zeros(51, np.float32), np.ones(51, np.float32), hidden_dim=32).eval()
        history = torch.rand(2, 12, 51, requires_grad=True)
        output = model(history, torch.tensor([.25, .75]))
        loss = output["expression"].square().mean() + output["jaw"].square().mean()
        loss.backward()
        assert output["expression"].shape == (2, 50)
        assert output["jaw"].shape == (2, 3)
        assert torch.isfinite(history.grad).all() and history.grad.norm() > 0
        print("PASS: artificial-mapper architecture/gradient test; NOT a pretrained Robot-FLAME demo.")


if __name__ == "__main__":
    main()
