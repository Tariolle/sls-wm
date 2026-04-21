"""Minimal YAML config loader with CLI override support.

Usage
-----
    import argparse
    from deepdash.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int)
    ...
    config = load_config("configs/e6.7-recon-cauchysls.yaml", parser.parse_args(), section="transformer")
    # config["lr"], config["batch_size"], etc.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config(
    yaml_path: str | Path,
    args: argparse.Namespace | None = None,
    section: str | None = None,
) -> dict[str, Any]:
    """Load config from YAML, optionally scoped to a section, with CLI overrides.

    Resolution order (later wins):
        1. ``model`` section (shared architecture defaults)
        2. ``section`` values (component-specific)
        3. CLI args that are not None (explicit user overrides)

    Parameters
    ----------
    yaml_path : str | Path
        Path to the YAML config file.
    args : argparse.Namespace, optional
        Parsed CLI arguments.  Only non-None values override the YAML.
    section : str, optional
        Top-level key to pull component-specific values from
        (e.g. "transformer", "fsq", "controller_ppo").

    Returns
    -------
    dict[str, Any]
        Merged configuration dictionary.
    """
    raw = load_yaml(yaml_path)

    # Start with shared model params
    config: dict[str, Any] = copy.deepcopy(raw.get("model", {}))

    # Layer on section-specific params
    if section and section in raw:
        config.update(raw[section])

    # CLI overrides (only explicitly provided values)
    if args is not None:
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value

    return config


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "e6.7-recon-cauchysls.yaml"


def apply_config(
    args: argparse.Namespace,
    section: str | None = None,
    config_path: str | Path | None = None,
) -> argparse.Namespace:
    """Fill ``None`` argparse values from YAML config.

    Call after ``parser.parse_args()`` with defaults set to ``None``
    for any argument that should be backed by the YAML.  Explicitly
    provided CLI args (non-None) always win.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    section : str, optional
        YAML section for component-specific values (e.g. "transformer").
    config_path : str | Path, optional
        Path to YAML config.  Falls back to ``args.config`` then
        ``configs/e6.7-recon-cauchysls.yaml``.
    """
    path = config_path or getattr(args, "config", None) or DEFAULT_CONFIG
    config = load_config(path, section=section)

    for key, value in config.items():
        arg_key = key.replace("-", "_")
        if not hasattr(args, arg_key) or getattr(args, arg_key) is None:
            setattr(args, arg_key, value)
    return args
