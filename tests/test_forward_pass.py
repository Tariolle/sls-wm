"""Deterministic forward-pass regression tests.

Limited to currently-live code paths. The V4-era TestFSQVAE / TestWorldModel /
TestMLPPolicy / TestLearnableGamma tests were retired 2026-04-26: vocab=1000
[8,5,5,5], multi-return WorldModel.forward, MLPPolicy and learnable_gamma are
no longer in any live config.

Run: pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def seed():
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


class TestDataSplit:
    def test_deterministic(self):
        from deepdash.data_split import get_val_episodes
        val1 = get_val_episodes("data/death_episodes", "data/expert_episodes")
        val2 = get_val_episodes("data/death_episodes", "data/expert_episodes")
        assert val1 == val2, "Data split not deterministic"

    def test_nonempty(self):
        from deepdash.data_split import get_val_episodes
        val = get_val_episodes("data/death_episodes", "data/expert_episodes")
        assert len(val) > 0, "Val set is empty"


class TestFSQMarginalUniformReg:
    def test_uniform_input_gives_near_zero_loss(self, device, seed):
        """Samples drawn exactly from Uniform[-half_d, +half_d] should
        produce a CvM statistic ~ O(1/M) which is tiny for large M.
        """
        from deepdash.fsq import fsq_marginal_uniform_reg
        torch.manual_seed(0)
        levels = [5, 5, 5, 5]
        half_levels = torch.tensor([L // 2 for L in levels],
                                    dtype=torch.float32, device=device)
        D = 4
        M = 10000
        # Draw U[-half, +half] per dim
        z = torch.empty(M, D, 1, 1, device=device)
        for d in range(D):
            z[:, d, 0, 0].uniform_(-float(half_levels[d]), float(half_levels[d]))
        loss = fsq_marginal_uniform_reg(z, half_levels)
        # CvM on ~10k uniform samples should be well below 1e-3
        assert loss.item() < 1e-3, f"uniform input gave CvM={loss.item()}"

    def test_concentrated_input_gives_high_loss(self, device, seed):
        """All-zero samples collapse the empirical CDF to a step at 0,
        far from Uniform[-half, +half]. CvM should be non-trivial.
        """
        from deepdash.fsq import fsq_marginal_uniform_reg
        levels = [5, 5, 5, 5]
        half_levels = torch.tensor([L // 2 for L in levels],
                                    dtype=torch.float32, device=device)
        D = 4
        M = 1000
        z = torch.zeros(M, D, 1, 1, device=device)
        loss = fsq_marginal_uniform_reg(z, half_levels)
        # Collapsed samples: CvM >= ~0.08 (analytic for step vs uniform)
        assert loss.item() > 0.05, f"collapsed input gave CvM={loss.item()}"

    def test_gradient_flows_through(self, device, seed):
        """Gradient w.r.t. z should be non-zero on non-uniform input."""
        from deepdash.fsq import fsq_marginal_uniform_reg
        torch.manual_seed(0)
        levels = [5, 5, 5, 5]
        half_levels = torch.tensor([L // 2 for L in levels],
                                    dtype=torch.float32, device=device)
        z = torch.randn(500, 4, 1, 1, device=device, requires_grad=True)
        loss = fsq_marginal_uniform_reg(z, half_levels)
        loss.backward()
        assert z.grad is not None
        assert z.grad.abs().sum().item() > 0
