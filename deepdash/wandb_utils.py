"""Weights & Biases integration with graceful fallback.

If wandb is not installed or not logged in, all calls are no-ops.
Training scripts can use this unconditionally.

Usage:
    from deepdash.wandb_utils import wandb_init, wandb_log, wandb_finish

    run = wandb_init(project="deepdash", name="transformer-512d", config=vars(args))
    for epoch in range(epochs):
        wandb_log({"train_loss": loss, "val_acc": acc, "epoch": epoch})
    wandb_finish()
"""

_run = None
_enabled = False


def wandb_init(project="deepdash", name=None, config=None, enabled=True,
               resume_id=None):
    """Initialize W&B run. Returns None if wandb unavailable.

    Args:
        resume_id: If set, resume an existing W&B run by its ID.
    """
    global _run, _enabled
    if not enabled:
        _enabled = False
        return None
    try:
        import wandb
        if config:
            import json
            config = json.loads(json.dumps(config, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
        kwargs = dict(project=project, name=name, config=config)
        if resume_id:
            kwargs["id"] = resume_id
            kwargs["resume"] = "allow"
        _run = wandb.init(**kwargs)
        _enabled = True
        print(f"W&B logging enabled: {_run.url}")
        return _run
    except Exception as e:
        print(f"W&B not available ({e}), logging to CSV only")
        _enabled = False
        return None


def wandb_run_id():
    """Return the current W&B run ID, or None."""
    if _enabled and _run is not None:
        return _run.id
    return None


def wandb_log(data, step=None):
    """Log metrics. No-op if wandb unavailable."""
    if _enabled and _run is not None:
        _run.log(data, step=step or data.get("iteration"))


def wandb_finish():
    """Finish W&B run. No-op if wandb unavailable."""
    global _run, _enabled
    if _enabled and _run is not None:
        _run.finish()
        _run = None
        _enabled = False
