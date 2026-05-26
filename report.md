# EEC 289A - Assignment 2: Inverted Pendulum World Model
**Kavin Rajasekaran**

---

## Task

The model is given 10 steps of real simulator data and has to predict the next 1000 steps without ever querying ground truth again. VPT80@0.25 is the main metric - it measures how many steps the prediction stays accurate on 80% of test windows before it drifts too far.

---

## Baseline

The starter model was a plain MLP predicting the next state delta. It got VPT80@0.25 around 21 on validation. nMSE would explode past step 20 on most windows, and OOD performance was much worse at around VPT 8. In the rollout video you can see the predicted pole drifting left while the cart follows it until it hits the boundary.

---

## What I Changed

### Three-Tier Physics Prior

The main architectural change was splitting the prediction into three components instead of having one MLP do everything:

1. **Hardcoded kinematic prior** - next_pos = pos + vel * dt. This is exact by Newton's laws, so the network never needs to learn it.
2. **Frozen linear residual** - a least-squares fit to the approximately linear LQR dynamics near equilibrium. Fitted once on the first training batch and frozen immediately, so there are no gradient updates and no drift.
3. **Bounded MLP residual** - a 3-layer residual network (hidden dim 256, SiLU, LayerNorm) that only learns the nonlinear correction. The output goes through tanh * 0.15, which means the MLP can refine predictions but can't override the priors entirely.

### Training Data

I switched from 110-step windows (data/dev) to 1010-step windows (data/public_scoreboard). The reasoning is simple: if the longest rollout the model ever sees during training is 100 steps, it has no reason to stay accurate at step 200.

### Rollout Training

Three things I added to handle compounding errors:

- **Horizon curriculum** - rollout length is sampled from 16 to 200 each update rather than fixed. This exposes the model to every timescale during training.
- **Scheduled sampling** - small Gaussian noise is injected into each predicted state before it gets fed back into the model. At eval time the model sees its own (imperfect) predictions as input, so this teaches it to handle that.
- **Input noise** - smaller noise on one-step inputs for OOD robustness.

Per-step loss weights increase linearly so later steps get more gradient signal. Individual step losses are clamped at 4.0 to avoid gradient explosion from early divergent rollouts. I also used Huber loss instead of MSE.

### Optimizer

I ended up at learning rate 3e-4, gradient clipping at 1.0, 20,000 updates, batch size 512, on an A100. At 5e-4 the training curve peaked around update 9000 and started oscillating - lowering it to 3e-4 smoothed things out and the best checkpoint came later in training.

---

## Results

| Run | Architecture | Test VPT80@0.25 | OOD VPT80@0.25 | nMSE@10 | nMSE@1000 |
|---|---|---|---|---|---|
| Starter baseline | Plain MLP | ~21 | ~8 | ~0.020 | - |
| First full run | MLP + priors | 23 | 23 | 0.0013 | 254 |
| **Final submission** | **MLP + priors** | **32** | **31** | **0.0006** | **255** |

Test and OOD ended up nearly the same (32 vs 31). Since OOD uses 3x higher initial noise that gap closing is a good sign - it's not just overfitting to the test distribution. Short-horizon accuracy improved a lot too: nMSE@10 dropped from 0.020 to 0.0006.

I also tried adding a GRU hidden state to give the model memory across rollout steps. The hope was that it could track accumulated error and self-correct. It didn't help - the GRU variant scored VPT=28, actually worse than the plain MLP - so I'm submitting the MLP checkpoint.

---

## Diagnostics

Looking at a sample of test windows from the final run: median VPT was 38, with a minimum of 27. The VPT80 of 32 is being pulled down by the hardest ~20% of windows, which fail somewhere in the 27-31 step range. Pole angle was the first state to fail in every single window - angular velocity accumulated roughly 100x more error than cart position over the full rollout. There's also a consistent negative bias across all four states that compounds over time, which explains the one-sided drift visible in the video.

---

## Summary

Going from VPT80@0.25 = 21 to 32 (test) / 31 (OOD) is about a 52% improvement, mostly from the physics priors, longer training windows, and the rollout curriculum. The gap to the top of the class (VPT around 39, nMSE@1000 = 0.28) is harder to close - without a correction mechanism, errors compound exponentially across steps and that's not something MLP tuning alone can fix.
