"""Global episode-level train/val split shared across all training scripts.

The split is deterministic (seed 42), stratified by source (death/expert),
and consistent across FSQ, Transformer, BC, and PPO. Val episodes are truly
held out: no model ever trains on their data.

Usage:
    from deepdash.data_split import get_val_episodes, is_val_episode

    val_set = get_val_episodes("data/death_episodes", "data/expert_episodes")
    if is_val_episode(ep_name, val_set):
        ...
"""

import re
from pathlib import Path

import numpy as np

SPLIT_SEED = 42
VAL_RATIO = 0.1


def get_val_episodes(death_dir="data/death_episodes",
                     expert_dir="data/expert_episodes"):
    """Compute the global val episode set.

    Stratified: 10% of death episodes + 10% of expert episodes.
    Uses base episode names (shift suffixes stripped).

    Returns:
        set of episode name strings that belong to val.
    """
    shift_re = re.compile(r"_s[+-]\d+_[+-]\d+$")
    rng = np.random.default_rng(SPLIT_SEED)

    val_episodes = set()

    for ep_dir in [death_dir, expert_dir]:
        ep_path = Path(ep_dir)
        if not ep_path.exists():
            continue
        # Collect base episode names (strip shift suffixes)
        base_names = sorted(set(
            shift_re.sub("", ep.name)
            for ep in ep_path.glob("*")
            if ep.is_dir() and (
                (ep / "frames.npy").exists() or (ep / "tokens.npy").exists()
            )
        ))
        if not base_names:
            continue
        idx = rng.permutation(len(base_names))
        val_count = max(1, int(len(base_names) * VAL_RATIO))
        val_episodes.update(base_names[i] for i in idx[:val_count])

    return val_episodes


def is_val_episode(ep_name, val_set):
    """Check if an episode belongs to val (handles shift suffixes)."""
    shift_re = re.compile(r"_s[+-]\d+_[+-]\d+$")
    base = shift_re.sub("", ep_name)
    return base in val_set
