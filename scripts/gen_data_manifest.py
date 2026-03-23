"""Generate SHA-256 manifest for all episode data files.

Creates a JSON manifest with checksums for every frames.npy and actions.npy
in the data directories. Used to detect data corruption or changes.

Usage:
    python scripts/gen_data_manifest.py
    python scripts/gen_data_manifest.py --verify  # check existing manifest
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate(data_dirs):
    manifest = {}
    for d in data_dirs:
        p = Path(d)
        if not p.exists():
            continue
        for ep in sorted(p.glob("*")):
            if not ep.is_dir():
                continue
            for fname in ["frames.npy", "actions.npy"]:
                fp = ep / fname
                if fp.exists():
                    key = str(fp.relative_to(p.parent.parent))
                    manifest[key] = sha256(fp)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate/verify data manifest")
    parser.add_argument("--death-dir", default="data/death_episodes")
    parser.add_argument("--expert-dir", default="data/expert_episodes")
    parser.add_argument("--output", default="data/manifest.json")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing manifest instead of generating")
    args = parser.parse_args()

    data_dirs = [args.death_dir, args.expert_dir]

    if args.verify:
        with open(args.output) as f:
            expected = json.load(f)
        actual = generate(data_dirs)
        ok, changed, missing, new = 0, 0, 0, 0
        for key, h in expected.items():
            if key not in actual:
                print(f"  MISSING: {key}")
                missing += 1
            elif actual[key] != h:
                print(f"  CHANGED: {key}")
                changed += 1
            else:
                ok += 1
        for key in actual:
            if key not in expected:
                print(f"  NEW: {key}")
                new += 1
        print(f"\n{ok} ok, {changed} changed, {missing} missing, {new} new")
        if changed or missing:
            sys.exit(1)
    else:
        manifest = generate(data_dirs)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(f"Manifest: {len(manifest)} files -> {args.output}")


if __name__ == "__main__":
    main()
