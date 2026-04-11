"""Merge local + A100 PPO W&B runs into one combined run.

Local run (m6m27o8s) already has A100-equivalent iterations (scaled).
A100 run (vhsgkej4) continues from the local checkpoint's raw iteration
count, so we remap its iterations to start after the local run.

Step 1 (--fetch): download both runs' histories to JSON.
Step 2 (--upload): create a new combined run.

Usage:
    python scripts/merge_wandb_ppo_runs.py --fetch
    python scripts/merge_wandb_ppo_runs.py --upload
"""

import argparse
import json
import math
from pathlib import Path


ENTITY = "florent-tariolle-insa-rouen-normandie"
PROJECT = "deepdash"
LOCAL_RUN_ID = "m6m27o8s"
A100_RUN_ID = "vhsgkej4"
CACHE = Path("checkpoints/wandb_ppo_merged.json")


def fetch():
    import wandb
    api = wandb.Api()

    def fetch_run(run_path):
        """Fetch and deduplicate a W&B run's history."""
        run = api.run(run_path)
        print(f"  {run.name} ({run.id})")
        # Merge rows with same iteration (train + eval logged separately)
        by_iter = {}
        for row in run.scan_history(page_size=10000):
            it = row.get("iteration")
            if it is None:
                continue
            if it not in by_iter:
                by_iter[it] = {}
            for k, v in row.items():
                if k.startswith("_"):
                    continue
                if v is None:
                    continue
                if isinstance(v, float) and math.isnan(v):
                    continue
                by_iter[it][k] = v
        rows = sorted(by_iter.values(), key=lambda r: r["iteration"])
        return run, rows

    # Fetch local run (already A100-scaled)
    print("Local run:")
    local_run, local_rows = fetch_run(f"{ENTITY}/{PROJECT}/{LOCAL_RUN_ID}")
    local_max_iter = local_rows[-1]["iteration"] if local_rows else 0
    print(f"  {len(local_rows)} rows, iterations {local_rows[0]['iteration']}-{local_max_iter}")

    # Fetch A100 run
    print("A100 run:")
    a100_run, a100_rows = fetch_run(f"{ENTITY}/{PROJECT}/{A100_RUN_ID}")
    a100_min_iter = a100_rows[0]["iteration"] if a100_rows else 0
    a100_max_iter = a100_rows[-1]["iteration"] if a100_rows else 0
    print(f"  {len(a100_rows)} rows, iterations {a100_min_iter}-{a100_max_iter}")

    # Remap A100 iterations to continue after local
    offset = local_max_iter - a100_min_iter + 1
    print(f"\nOffset for A100: {offset} (A100 iter {a100_min_iter} -> {a100_min_iter + offset})")
    for row in a100_rows:
        row["iteration"] += offset

    combined = local_rows + a100_rows
    combined.sort(key=lambda r: r["iteration"])
    print(f"Combined: {len(combined)} rows, iterations {combined[0]['iteration']}-{combined[-1]['iteration']}")

    # Save config from A100 run (most recent)
    data = {
        "rows": combined,
        "config": dict(a100_run.config),
        "local_max_iter": local_max_iter,
        "a100_offset": offset,
    }
    CACHE.write_text(json.dumps(data))
    print(f"Saved to {CACHE}")


def upload():
    import wandb
    data = json.loads(CACHE.read_text())
    rows = data["rows"]
    config = data["config"]
    notes = f"Combined: local ({LOCAL_RUN_ID}, iters 1-{data['local_max_iter']}) + A100 ({A100_RUN_ID}, offset {data['a100_offset']})"
    print(f"Uploading {len(rows)} rows...")

    wandb.init(
        project=PROJECT,
        name="ppo-512d-combined",
        config=config,
        notes=notes,
    )
    for i, row in enumerate(rows):
        wandb.log(row)
        if (i + 1) % 1000 == 0:
            print(f"  logged {i + 1}/{len(rows)}")
    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        fetch()
    elif args.upload:
        upload()
    else:
        print("Use --fetch or --upload")
