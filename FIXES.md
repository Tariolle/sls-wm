# Deferred Fixes

Bugs and inconsistencies that cannot be fixed while PPO training is running
on the supercomputer, because they would break checkpoint compatibility or
change training dynamics mid-run.

## After PPO run completes

### 1. MTP targets off-by-one (train_controller_ppo.py)

The multi-token prediction auxiliary loss predicts actions starting from
the current step (t, t+1, ..., t+7) instead of future steps (t+1, ..., t+8).

**File:** `scripts/train_controller_ppo.py`, around line 250 (after dead code removal)

```python
# CURRENT (wrong): mtp_targets[t, :, k] = actions_seq[t + k]
for k in range(L):
    end = T - k
    if end > 0:
        mtp_targets[:end, :, k] = actions_seq[k:k + end]

# FIX: shift by +1 so k=0 predicts next action, not current
for k in range(L):
    src = k + 1
    if src < T:
        length = T - src
        mtp_targets[:length, :, k] = actions_seq[src:src + length]
```

**Impact:** Low-to-moderate. MTP is an auxiliary loss (weight 0.1), and the
off-by-one only shifts predictions by one step. The first slot trains on
same-step actions (which the controller already chose), providing no
lookahead signal for that slot.

### 2. Val loss computed outside AMP (train_transformer.py)

In `val_epoch`, `focal_cross_entropy` is called outside `torch.autocast`,
so val loss is computed in fp32. In `train_epoch`, it runs inside autocast
(fp16). This causes a subtle precision mismatch in logged metrics.

**File:** `scripts/train_transformer.py`, `val_epoch` function

**Fix:** Move `focal_cross_entropy` call inside the `torch.autocast` block
to match `train_epoch`.

**Impact:** Cosmetic. Only affects reported metrics, not model weights.
The gap metric (val - train) includes a precision artifact.

## Before next full retrain (if time permits before model freeze)

### 3. Paper Equation 2 vs code: SLS dim_weights

The paper defines weighted distance as:
  d^2(i,j) = sum_d alpha_d * (c_d(i) - c_d(j))^2

The code multiplies weights BEFORE squaring:
  diff = diff * w           -->  w_d * delta_d
  sq_dist = (diff**2).sum() -->  sum_d w_d^2 * delta_d^2

So the code effectively uses alpha^2, not alpha.

**Two valid resolutions:**
- **(A) Fix the code:** `sq_dist = ((diff ** 2) * w.view(1, 1, -1)).sum(dim=-1)`
  Requires retraining transformer, BC, and PPO from scratch.
- **(B) Fix the paper equation:** Change to `d^2(i,j) = sum_d (alpha_d * delta_d)^2`.
  This is still a valid weighted distance metric. Document that alpha weights
  are applied as scaling factors on coordinate differences, not on squared
  differences. The calibrated values [1.29, 0.85, 0.97, 0.89] are correct
  for this formulation.

Option B is recommended if there is no time to retrain before model freeze.

## Housekeeping (anytime)

### 4. Untrack profiler_trace.json

Already covered by `*.json` in `.gitignore` but was committed before that
rule existed. Run:
```
git rm --cached profiler_trace.json
```

### 5. Unused controller classes

`Controller` (CMA-ES), `PolicyController`, and `TransformerPolicy` in
`deepdash/controller.py` are V0-V3 legacy. Not used by any current script.
Consider removing or moving to an archive module after model freeze.
