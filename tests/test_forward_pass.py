"""Deterministic forward pass regression tests.

Verify that model forward passes produce identical outputs given fixed
seeds and inputs. Catches silent regressions when refactoring.

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


class TestFSQVAE:
    def test_encode_deterministic(self, device, seed):
        from deepdash.fsq import FSQVAE
        model = FSQVAE(levels=[8, 5, 5, 5]).to(device).eval()
        x = torch.randn(1, 1, 64, 64, device=device)
        with torch.no_grad():
            idx1 = model.encode(x)
            idx2 = model.encode(x)
        assert torch.equal(idx1, idx2), "FSQ encode not deterministic"

    def test_encode_shape(self, device, seed):
        from deepdash.fsq import FSQVAE
        model = FSQVAE(levels=[8, 5, 5, 5]).to(device).eval()
        x = torch.randn(2, 1, 64, 64, device=device)
        with torch.no_grad():
            idx = model.encode(x)
        assert idx.shape == (2, 8, 8), f"Expected (2, 8, 8), got {idx.shape}"

    def test_roundtrip(self, device, seed):
        from deepdash.fsq import FSQVAE
        model = FSQVAE(levels=[8, 5, 5, 5]).to(device).eval()
        x = torch.randn(1, 1, 64, 64, device=device)
        with torch.no_grad():
            recon, _, _ = model(x)
        assert recon.shape == x.shape

    def test_codebook_range(self, device, seed):
        from deepdash.fsq import FSQVAE
        model = FSQVAE(levels=[8, 5, 5, 5]).to(device).eval()
        x = torch.randn(4, 1, 64, 64, device=device)
        with torch.no_grad():
            idx = model.encode(x)
        assert idx.min() >= 0
        assert idx.max() < 1000


class TestWorldModel:
    """Tests use V4 architecture: 512d, 8 heads, 8 layers, AdaLN."""

    def _make_model(self, device):
        from deepdash.world_model import WorldModel
        return WorldModel(
            vocab_size=1000, embed_dim=512, n_heads=8, n_layers=8,
            context_frames=4, tokens_per_frame=64, adaln=True,
        ).to(device).eval()

    def _make_inputs(self, device, B=1):
        tokens = torch.randint(0, 1000, (B, 5, 65), device=device)
        tokens[:, :, -1] = 1000  # ALIVE token
        actions = torch.randint(0, 2, (B, 4), device=device)
        return tokens, actions

    def test_forward_deterministic(self, device, seed):
        model = self._make_model(device)
        tokens, actions = self._make_inputs(device)
        with torch.no_grad():
            logits1, cpc1 = model(tokens, actions)
            logits2, cpc2 = model(tokens, actions)
        assert torch.allclose(logits1, logits2, atol=1e-5), "Forward not deterministic"

    def test_forward_shape(self, device, seed):
        model = self._make_model(device)
        tokens, actions = self._make_inputs(device, B=2)
        with torch.no_grad():
            logits, cpc = model(tokens, actions)
        assert logits.shape == (2, 65, 1002), f"Expected (2, 65, 1002), got {logits.shape}"

    def test_encode_context_shape(self, device, seed):
        model = self._make_model(device)
        tokens = torch.randint(0, 1000, (1, 4, 65), device=device)
        tokens[:, :, -1] = 1000
        actions = torch.randint(0, 2, (1, 4), device=device)
        with torch.no_grad():
            h_t = model.encode_context(tokens, actions)
        assert h_t.shape == (1, 512), f"Expected (1, 512), got {h_t.shape}"

    def test_predict_next_frame_shape(self, device, seed):
        model = self._make_model(device)
        tokens = torch.randint(0, 1000, (1, 4, 65), device=device)
        tokens[:, :, -1] = 1000
        actions = torch.randint(0, 2, (1, 4), device=device)
        with torch.no_grad():
            pred_tokens, death_prob, h_t = model.predict_next_frame(
                tokens, actions, temperature=0.0, return_hidden=True)
        assert pred_tokens.shape == (1, 64)
        assert death_prob.shape == (1,)
        assert h_t.shape == (1, 512)


class TestCNNPolicy:
    """Tests use V4 architecture: h_dim=512 matching world model embed_dim."""

    def test_forward_shape(self, device, seed):
        from deepdash.controller import CNNPolicy
        policy = CNNPolicy(vocab_size=1000, h_dim=512).to(device).eval()
        tokens = torch.randint(0, 1000, (2, 64), device=device)
        h_t = torch.randn(2, 512, device=device)
        with torch.no_grad():
            prob, value = policy(tokens, h_t)
        assert prob.shape == (2,)
        assert value.shape == (2,)
        assert (prob >= 0).all() and (prob <= 1).all()

    def test_mtp_shape(self, device, seed):
        from deepdash.controller import CNNPolicy
        policy = CNNPolicy(vocab_size=1000, h_dim=512, mtp_steps=8).to(device).eval()
        tokens = torch.randint(0, 1000, (2, 64), device=device)
        h_t = torch.randn(2, 512, device=device)
        with torch.no_grad():
            mtp = policy.predict_future_actions(tokens, h_t)
        assert mtp.shape == (2, 8)

    def test_act_deterministic(self, device, seed):
        from deepdash.controller import CNNPolicy
        policy = CNNPolicy(vocab_size=1000, h_dim=512).to(device).eval()
        tokens = torch.randint(0, 1000, (1, 64), device=device)
        h_t = torch.randn(1, 512, device=device)
        with torch.no_grad():
            a1 = policy.act_deterministic(tokens, h_t)
            a2 = policy.act_deterministic(tokens, h_t)
        assert torch.equal(a1, a2)


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
