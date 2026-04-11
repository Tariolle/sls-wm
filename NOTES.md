# Notes for Next Paper Draft Update

## Controller Scaling Findings (2026-04-11)

### Setup
- Old controller: 50K params, linear heads on CNN+h_t features
- New controller (V4): 966K params, 3-layer MLP(512) trunk with LayerNorm+SiLU
- Both trained on A100 with 512 episodes/iter

### Results
- Old controllers plateaued around 7k iterations at eval/survival ~29.8 and ~30.5
- New controller plateaus around 20k iterations at eval/survival ~31.85
- Gain: +1.35 to +2.05 survival steps, with 20x more params and 3x longer convergence

### Qualitative observations (real game deployment)
- New controller handles patterns it previously could not (smarter)
- Probability distributions are softer: more time in 0.3-0.7 range vs old controller's binary behavior
- Hypothesis: 3-layer MLP with LayerNorm learns smoother decision boundaries vs old linear heads that were forced into sharp logits
- Inference stays at 30fps, so timing/latency is not a factor

### Dream play test (human vs controller)
- Human consistently beats the V4 controller in dreams
- Controller performance is inconsistent: sometimes handles hard obstacles well, sometimes dies on trivial spikes (fails to jump at all)
- This matches the soft probability distributions seen in deploy logs (too much time in 0.3-0.7 range), causing stochastic sampling to randomly miss obvious jumps
- Conclusion: bottleneck is still controller, not world model quality
- The diminishing returns curve suggests the current MLP architecture is struggling to close the remaining gap with more iterations alone

### Cube vs spaceship forms
- Levels are roughly 2/3 cube, 1/3 spaceship. Levels start in cube form.
- Dream eval (29.8, 30.5, 31.85) uses the same episode mix across all controllers, so comparison is valid
- Real-game zero-shot tests never reached spaceship form (starts at ~frame 30) because controllers die before that in cube
- Spaceship performance in dreams is decent but has never been validated in real game
- Per-form dream metrics would still be useful to understand where gains come from

### Core diagnosis: memorization, not generalization (checkpoint mode test)
- Tested with checkpoint mode (restart 20 frames before death instead of level start)
- Controller clears 20% of the level on complex patterns, dies randomly on trivial obstacles
- Same behavior in spaceship form: survives hard moments, dies on easy ones
- Timing/reflexes are solved (consistently passes triple spikes, frame-perfect)
- ALL failures stem from decision ("should I jump") never from reflex ("can I jump in time")
- Deploy uses hard threshold (p > 0.5), not sampling, so soft probabilities are not the cause
- The controller doesn't generalize: the same obstacle type at different y-positions or with different surrounding tokens produces different decisions
- It memorizes specific dream token grids rather than learning position-invariant obstacle features
- No amount of PPO iterations, entropy tuning, or MLP depth will fix a representation problem

### Open questions
- What input representation would support position-invariant obstacle detection?
- Would relative spatial features (player-to-obstacle distance) help vs raw token grid?
- Can the transformer h_t be improved to encode obstacle-relevant features more explicitly?
- How to cheaply classify episodes as cube vs spaceship for per-form metrics?
