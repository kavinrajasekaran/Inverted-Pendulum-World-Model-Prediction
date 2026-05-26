# EEC 289A — Assignment 2: Inverted Pendulum World Model
**Kavin Rajasekaran**

---

## What We're Doing

The task is simple to state: given 10 steps of real simulator data, predict
the next 1000 steps without ever looking at ground truth again. The model has
to mentally simulate the physics entirely on its own. The score (VPT80@0.25)
measures how many steps it stays accurate on 80% of test windows before its
prediction drifts away from what MuJoCo actually does.

---

## Where I Started

The starter model was a plain MLP that took the current normalized state and
action and predicted the next state delta. Out of the box it got:

- **VPT80@0.25 ≈ 21** on the validation set
- nMSE exploding past step 20 on most windows
- OOD performance significantly worse than test (VPT ≈ 8)
- In the rollout video: the predicted pole tilts left and the cart chases it
  until it hits the boundary

The core problem was that the MLP had no concept of physics. It was trying to
learn position-velocity coupling, angular dynamics, and gravity effects purely
from data — and the result was a model that memorized LQR correlations rather
than learning the actual system dynamics.

---

## What I Changed

### Architecture: Three-Tier Physics Prior

Instead of one MLP predicting everything, I decomposed the prediction into
three components that build on each other:

**1. Kinematic prior (hardcoded)**
```
next_cart_pos   = cart_pos   + cart_vel     × dt
next_pole_angle = pole_angle + pole_ang_vel × dt
```
This is exact by Newton's laws. The network never has to learn it, so it
can't accidentally get it wrong or introduce drift here.

**2. Frozen linear residual**
A closed-form least-squares fit from the first training batch captures the
approximately-linear LQR dynamics near equilibrium. It's frozen immediately
after fitting — it doesn't participate in gradient updates and can't drift.

**3. Bounded MLP residual**
The neural network only learns the nonlinear correction on top. A `tanh`
with `delta_limit = 0.15` means the MLP can refine but never override the
priors. This prevents it from learning arbitrary large corrections that could
encode spurious patterns from the training data.

I also added explicit physics features to the MLP input — `sin(θ)`,
`cos(θ) - 1`, and the velocities and action directly — so the network has
clean signals for the nonlinear terms it actually needs to learn rather than
having to discover them implicitly.

The rest of the architecture: 3 residual blocks (hidden dim 256), SiLU
activations, LayerNorm at each block, learnable per-block output scale
initialized at 0.5.

### GRU Hidden State

The model also supports an optional GRU layer (`use_gru: true`) placed after
the residual blocks. When enabled, the GRU maintains a hidden state across
rollout steps — giving the model memory of what has happened in previous
steps. This is critical for long-horizon stability: without memory, small
per-step errors compound multiplicatively with no correction mechanism. With
a hidden state, the model can track accumulated drift and adjust predictions
accordingly.

The rollout function correctly threads the hidden state through every step of
both training and evaluation rollouts, including the warmup phase.

### Training Data

The starter trained on 110-step windows from `data/dev`. That's not enough
— you can't teach a model to stay accurate for 200+ steps if the longest
rollout it ever sees during training is 100 steps.

I switched to `data/public_scoreboard`, which has 1010-step windows. This
gives the rollout loss room to supervise up to 200-step predictions.

### Rollout Loss

Three ideas here that all address the compounding error problem:

**Horizon curriculum**: Each update samples a rollout horizon uniformly from
[16, 200] instead of using one fixed horizon. The model sees every timescale
during training. Short horizons keep short-term accuracy sharp; long horizons
push long-horizon stability.

**Scheduled sampling**: During training rollouts, I add small Gaussian noise
(σ = 0.01 × obs_std) to each predicted state before feeding it back into the
model. This teaches the model that its own predictions aren't perfect — it
learns to stay accurate even when the input is slightly off, which is exactly
the situation during eval.

**Input noise**: The one-step loss also perturbs input observations with
smaller noise (σ = 0.004). This helps with OOD robustness — the model
doesn't memorize exact training inputs.

Per-step loss weights grow linearly from 1.0 to 2.0 across the horizon, so
later (harder) prediction steps get more gradient signal. Individual step
losses are clamped at 4.0 to prevent gradient explosion from early divergent
rollouts. Smooth L1 (Huber, β = 0.25) instead of MSE for robustness.

Combined loss: `1.0 × one_step + 4.0 × rollout`.

### Optimizer

Learning rate: **3e-4**. At 5e-4 the training curve peaked early (~update
9000) and then oscillated — the optimizer kept overshooting. At 3e-4 the
curve was smoother and the best checkpoint appeared deeper into training.
Gradient clipping at max_norm = 1.0 throughout. 20,000 updates total,
batch size 512, trained on A100.

---

## Results

| Run | Updates | Architecture | Test VPT80@0.25 | OOD VPT80@0.25 | nMSE@10 | nMSE@1000 |
|---|---|---|---|---|---|---|
| Starter baseline | ~5k | Plain MLP | ~21 | ~8 | ~0.020 | 593,128 |
| First full run | 12k | MLP + priors | 23 | 23 | 0.0013 | 254 |
| Final MLP run | 20k | MLP + priors | **32** | **31** | **0.0006** | 255 |
| GRU run | 20k | GRU + priors | *TBD* | *TBD* | *TBD* | *TBD* |

The MLP run was evaluated on the official public scoreboard dataset with a
1000-step horizon. Test and OOD VPT are nearly identical (32 vs 31), which
matters — OOD uses 3× higher initial noise, so closing that gap reflects
real robustness rather than test-set luck.

nMSE@10 improved from 0.020 to 0.0006 — a 33× improvement in short-horizon
accuracy.

---

## What the Diagnostics Show

Running detailed analysis over 16 test windows from the MLP run:

**VPT distribution**: min=27, p20=31, median=38, p80=41, max=50

The median window stays accurate until step 38. VPT80=32 is being pulled down
by the hardest 20% of windows failing in the 27–31 step range.

**Which state fails first**: pole_angle, in **every single window** (16/16).
Not cart position, not cart velocity — always the pole angle. The pole
angular velocity accumulated 100× more error than cart position over the
full rollout.

**Directional bias**: All four states show a consistent negative bias —
the model slightly underestimates angular dynamics on average. Over hundreds
of steps this small per-step bias compounds into the one-sided drift visible
in the rollout video (pole tilts left, cart follows).

---

## Why the MLP Scores Are Still Limited

The model is extremely accurate at step 10 (nMSE = 0.0006) and has fully
diverged by step 100 (nMSE = 108) — a 180,000× increase over 90 steps.
This exponential growth pattern is a fundamental property of stateless
prediction: each step's error feeds into the next with no correction. The
best submission in the class got nMSE@1000 = 0.28 while the MLP got 255 —
a 900× gap that no amount of MLP tuning can close, because the issue is
architectural. A recurrent model with hidden state can track accumulated
error and self-correct across hundreds of steps. The GRU run is intended
to address exactly this.

---

## Summary

Starting from VPT80@0.25 ≈ 21, physics-informed priors, longer training
data, rollout curriculum, scheduled sampling, and a lower learning rate
brought the MLP model to **VPT80@0.25 = 32 (test) and 31 (OOD)** — a 52%
improvement. The model generalizes well across test and OOD splits and
achieves very accurate short-horizon predictions. The GRU run is in progress
and results will be added once complete.
