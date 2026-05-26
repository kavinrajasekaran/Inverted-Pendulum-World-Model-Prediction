# EEC 289A — Assignment 2: Inverted Pendulum World Model
**Kavin Rajasekaran**

---

## Task

Given 10 steps of real simulator data, predict the next 1000 steps without
ever querying ground truth again. The score (VPT80@0.25) measures how many
steps the prediction stays accurate on 80% of test windows.

---

## Baseline

The starter model was a plain MLP predicting the next state delta. It
achieved VPT80@0.25 ≈ 21 on validation, with nMSE exploding past step 20 and
OOD performance collapsing to VPT ≈ 8. The rollout video shows the predicted
pole drifting left until the cart hits the boundary.

---

## What I Changed

### Three-Tier Physics Prior

I decomposed the prediction into three stacked components:

1. **Hardcoded kinematic prior** — `next_pos = pos + vel × dt`. Exact by
   Newton's laws; the network never has to learn or approximate this.
2. **Frozen linear residual** — a least-squares fit to the approximately-linear
   LQR dynamics near equilibrium, fitted once and frozen. No gradient drift.
3. **Bounded MLP residual** — a 3-layer residual network (hidden dim 256,
   SiLU, LayerNorm) learning only the nonlinear correction. Output is clamped
   by `tanh × 0.15` so the MLP can refine but never override the priors.

### Longer Training Data

Switched from 110-step windows (`data/dev`) to 1010-step windows
(`data/public_scoreboard`), giving the rollout loss room to supervise
predictions up to 200 steps ahead.

### Rollout Training Improvements

- **Horizon curriculum**: rollout length sampled uniformly from [16, 200] each
  update, exposing the model to every timescale.
- **Scheduled sampling**: Gaussian noise (σ = 0.01 × obs\_std) injected into
  each predicted state during rollout, making the model robust to its own
  errors.
- **Input noise**: smaller noise (σ = 0.004) on one-step inputs for OOD
  robustness.
- Per-step loss weights increase linearly, step losses clamped at 4.0, Huber
  loss (β = 0.25).

### Optimizer

Learning rate 3e-4, gradient clipping at 1.0, 20,000 updates, batch size 512,
trained on A100.

---

## Results

| Run | Architecture | Test VPT80@0.25 | OOD VPT80@0.25 | nMSE@10 | nMSE@1000 |
|---|---|---|---|---|---|
| Starter baseline | Plain MLP | ~21 | ~8 | ~0.020 | 593,128 |
| First full run | MLP + priors | 23 | 23 | 0.0013 | 254 |
| **Final submission** | **MLP + priors** | **32** | **31** | **0.0006** | **255** |

Test and OOD VPT are nearly identical (32 vs 31), indicating genuine
robustness rather than test-set luck. Short-horizon accuracy improved 33×
(nMSE@10: 0.020 → 0.0006).

I also experimented with adding a GRU hidden state to carry memory across
rollout steps, hoping it would reduce long-horizon error compounding. It
didn't — the GRU variant scored VPT=28, worse than the plain MLP, so the MLP
checkpoint is submitted.

---

## Diagnostics

Over 16 test windows: median VPT = 38, min = 27. VPT80 = 32 is pulled down
by the hardest ~20% of windows. Pole angle fails first in every window (16/16)
— angular velocity accumulates ~100× more error than cart position. All states
show a slight negative bias that compounds into the one-sided drift visible in
the rollout video.

---

## Summary

Physics priors, longer training windows, rollout curriculum, and scheduled
sampling improved the model from VPT80@0.25 = 21 to **32 (test) / 31 (OOD)**
— a 52% gain. The remaining gap to the top of the class (VPT ≈ 39,
nMSE@1000 = 0.28) reflects the fundamental limitation of stateless prediction:
without a correction mechanism, per-step errors compound exponentially and no
amount of MLP tuning closes that gap.
