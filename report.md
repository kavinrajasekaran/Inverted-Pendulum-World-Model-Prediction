# EEC 289A - Assignment 2: Inverted Pendulum World Model
**Kavin Rajasekaran**

---

## Task

Given 10 steps of real simulator data, predict the next 1000 steps without ever querying ground truth again. The score (VPT80@0.25) measures how many steps the prediction stays accurate on 80% of test windows.

---

## Baseline

The starter model was a plain MLP predicting the next state delta. It got VPT80@0.25 around 21 on validation, with nMSE exploding past step 20 and OOD performance collapsing to around VPT 8. In the rollout video, the predicted pole drifts left until the cart hits the boundary.

---

## What I Changed

### Three-Tier Physics Prior

Instead of one MLP doing everything, I split the prediction into three stacked components:

1. **Hardcoded kinematic prior** - next_pos = pos + vel * dt. This is exact by Newton's laws, so the network never has to learn or approximate it.
2. **Frozen linear residual** - a least-squares fit to the approximately linear LQR dynamics near equilibrium, fitted once on the first training batch and then frozen. No gradient updates, no drift.
3. **Bounded MLP residual** - a 3-layer residual network (hidden dim 256, SiLU activations, LayerNorm) that only learns the nonlinear correction on top. The output is clamped by tanh * 0.15, so the MLP can refine but never override the priors.

### Longer Training Data

I switched from 110-step windows (data/dev) to 1010-step windows (data/public_scoreboard). You can't teach a model to stay accurate for 200+ steps if the longest rollout it ever sees during training is 100 steps.

### Rollout Training

A few things I added to deal with compounding errors:

- **Horizon curriculum** - rollout length is sampled uniformly from 16 to 200 each update, so the model sees every timescale during training.
- **Scheduled sampling** - small Gaussian noise is added to each predicted state during rollout. This teaches the model to stay accurate even when its own input is slightly off, which is exactly what happens at eval time.
- **Input noise** - smaller noise on one-step inputs helps with OOD robustness.

Per-step loss weights increase linearly across the horizon so later steps get more gradient signal. Step losses are clamped at 4.0 to avoid gradient explosion from early divergent rollouts. Huber loss instead of MSE for robustness.

### Optimizer

Learning rate 3e-4, gradient clipping at 1.0, 20,000 updates, batch size 512, trained on A100. At 5e-4 the training curve peaked early and oscillated. At 3e-4 it was smoother and the best checkpoint came later.

---

## Results

| Run | Architecture | Test VPT80@0.25 | OOD VPT80@0.25 | nMSE@10 | nMSE@1000 |
|---|---|---|---|---|---|
| Starter baseline | Plain MLP | ~21 | ~8 | ~0.020 | 593,128 |
| First full run | MLP + priors | 23 | 23 | 0.0013 | 254 |
| **Final submission** | **MLP + priors** | **32** | **31** | **0.0006** | **255** |

Test and OOD VPT are nearly identical (32 vs 31), which I think reflects real robustness - OOD uses 3x higher initial noise. Short-horizon accuracy improved a lot: nMSE@10 went from 0.020 down to 0.0006.

I also tried adding a GRU hidden state to carry memory across rollout steps, hoping it would help with long-horizon error compounding. It actually made things worse - the GRU variant got VPT=28, lower than the plain MLP - so the MLP checkpoint is what I'm submitting.

---

## Diagnostics

Over 16 test windows: median VPT was 38, min was 27. The VPT80 of 32 is being pulled down by the hardest ~20% of windows failing in the 27-31 step range. Pole angle fails first in every single window (16/16) - angular velocity accumulated about 100x more error than cart position. All four states have a slight negative bias that compounds into the one-sided drift you can see in the rollout video.

---

## Summary

Physics priors, longer training windows, rollout curriculum, and scheduled sampling improved the model from VPT80@0.25 = 21 to 32 (test) / 31 (OOD) - a 52% improvement. The remaining gap to the top of the class (VPT around 39, nMSE@1000 = 0.28) is mostly architectural - without any correction mechanism, per-step errors compound exponentially and MLP tuning alone can't fix that.
