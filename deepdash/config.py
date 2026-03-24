"""Minimal YAML config loader with CLI override support.

Usage
-----
    import argparse
    from deepdash.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int)
    ...
    config = load_config("configs/v3.yaml", parser.parse_args(), section="transformer")
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
