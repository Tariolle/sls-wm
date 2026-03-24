"""Diagnostic: does the Transformer react to different actions?

Feeds the same context frames with all-idle vs all-jump actions and
measures how much the predicted tokens differ.

Usage:
    python scripts/diagnose_action_sensitivity.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepdash.world_model import WorldModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = WorldModel(
        vocab_size=1000, embed_dim=384, n_heads=8, n_layers=8,
        context_frames=4, tokens_per_frame=64,
    ).to(device)
    state = torch.load("checkpoints/transformer_best.pt",
                       map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # 1. Action embedding similarity
    e0 = model.action_embed.weight[0]
    e1 = model.action_embed.weight[1]
    cos_sim = torch.nn.functional.cosine_similarity(
        e0.unsqueeze(0), e1.unsqueeze(0)).item()
    l2_dist = (e0 - e1).norm().item()
    print(f"\nAction embeddings:")
    print(f"  cosine similarity: {cos_sim:.4f}  (1.0 = identical)")
    print(f"  L2 distance:       {l2_dist:.4f}")

    # 2. Action distribution in dataset
    episodes_dir = Path("data/death_episodes")
    idle_count, jump_count = 0, 0
    for ep in sorted(episodes_dir.glob("*")):
        ap = ep / "actions.npy"
        if ap.exists():
            a = np.load(ap)
            idle_count += int((a == 0).sum())
            jump_count += int((a == 1).sum())
    total = idle_count + jump_count
    print(f"\nAction distribution:")
    print(f"  idle: {idle_count} ({100 * idle_count / total:.1f}%)")
    print(f"  jump: {jump_count} ({100 * jump_count / total:.1f}%)")

    # 3. Same context, idle vs jump -> compare predictions
    eps = [ep for ep in sorted(episodes_dir.glob("*"))
           if (ep / "tokens.npy").exists()]
    print(f"\nTokenized episodes available: {len(eps)}")

    actions_idle = torch.zeros(1, 4, dtype=torch.long, device=device)
    actions_jump = torch.ones(1, 4, dtype=torch.long, device=device)

    diffs = []
    logit_diffs = []
    n_test = min(200, len(eps))

    with torch.no_grad():
        for ep in eps[:n_test]:
            t = np.load(ep / "tokens.npy").astype(np.int64)
            if len(t) < 5:
                continue
            ctx = torch.from_numpy(t[:4]).unsqueeze(0).to(device)
            status = torch.full((1, 4, 1), model.ALIVE_TOKEN,
                                dtype=torch.long, device=device)
            ctx_s = torch.cat([ctx, status], dim=2)

            target = torch.from_numpy(t[4:5]).unsqueeze(0).to(device)
            st = torch.full((1, 1, 1), model.ALIVE_TOKEN,
                            dtype=torch.long, device=device)
            target_s = torch.cat([target, st], dim=2)

            inp = torch.cat([ctx_s, target_s], dim=1)

            li, _, _ = model(inp, actions_idle)
            lj, _, _ = model(inp, actions_jump)

            preds_i = li[0, :64].argmax(dim=-1)
            preds_j = lj[0, :64].argmax(dim=-1)
            diffs.append((preds_i != preds_j).sum().item())
            logit_diffs.append((li[0, :64] - lj[0, :64]).abs().mean().item())

    print(f"\nAction sensitivity ({len(diffs)} episodes tested):")
    print(f"  mean tokens different:  {np.mean(diffs):.1f}/64")
    print(f"  min/max tokens diff:    {min(diffs)}/{max(diffs)}")
    print(f"  episodes with 0 diff:   {sum(1 for d in diffs if d == 0)}/{len(diffs)}")
    print(f"  mean |logit diff|:      {np.mean(logit_diffs):.6f}")

    if np.mean(diffs) < 1.0:
        print("\n⚠ Le Transformer ignore quasiment l'action. "
              "Le controleur ne peut pas apprendre.")
    elif np.mean(diffs) < 5.0:
        print("\n⚠ Sensibilite faible. Le Transformer reagit un peu "
              "mais le signal est tenu.")
    else:
        print("\n✓ Le Transformer reagit significativement aux actions.")


if __name__ == "__main__":
    main()
