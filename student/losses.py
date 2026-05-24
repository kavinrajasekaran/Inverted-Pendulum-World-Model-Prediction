"""Student training loss: one-step + multi-horizon rollout with symmetry augmentation.

Three things this loss does differently from the v1 starter:

  * **Reflection augmentation.**  The InvertedPendulum dynamics are
    left-right symmetric in (state, action).  For every batch we randomly
    reflect ~50% of the windows (state, action, next_state) -> (-state,
    -action, -next_state).  Combined with the bias-free linear prior in
    model.py, this prevents the NN from picking up an asymmetric drift from
    the finite LQR-controlled training set (which was the root cause of
    the "cart drifts left forever" failure in v1 rollouts).

  * **Horizon curriculum.**  Every update samples a rollout horizon uniformly
    in [rollout_min_horizon, rollout_train_horizon] so the model is forced
    to be accurate at *every* horizon length rather than memorising a single
    rollout depth.

  * **Scheduled-sampling noise.**  Small Gaussian noise is injected on every
    rolled-out state (scaled by observation std), teaching the model to
    recover from realistic accumulated error instead of amplifying it.

The smooth-L1 (Huber) per-step loss keeps gradients bounded if a rollout
does start to diverge, and per-step weights emphasise late steps so that
long-horizon stability gets most of the training pressure.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout


def _sync_model_normalizer(model, normalizer) -> None:
    setter = getattr(model, "set_normalizer", None)
    if setter is not None:
        setter(normalizer)


def _initialize_model(model, states: torch.Tensor, actions: torch.Tensor) -> None:
    initializer = getattr(model, "initialize_linear_dynamics", None)
    if initializer is not None:
        initializer(states, actions)


def _reflect_augment(
    states: torch.Tensor,
    actions: torch.Tensor,
    flip_probability: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly negate (state, action) for ~`flip_probability` fraction of windows.

    Because the cart-pole dynamics is exactly antisymmetric in (s, a), the
    next state is also negated, so the (state, action, next_state) triple
    remains a valid sample from the true dynamics.  No information is lost,
    and the model is forced to learn symmetric dynamics on average.
    """
    if flip_probability <= 0.0:
        return states, actions
    batch_size = states.shape[0]
    flip = (
        torch.rand(batch_size, device=states.device) < float(flip_probability)
    ).to(states.dtype)
    sign_states = (1.0 - 2.0 * flip).view(batch_size, 1, 1)
    sign_actions = sign_states  # actions get the same sign as the state under reflection
    return states * sign_states, actions * sign_actions


def one_step_delta_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    *,
    obs_noise_sigma: float = 0.0,
) -> torch.Tensor:
    obs = states[:, :-1].reshape(-1, states.shape[-1])
    act = actions.reshape(-1, actions.shape[-1])
    next_obs = states[:, 1:].reshape(-1, states.shape[-1])
    obs_norm = normalizer.normalize_obs(obs)
    act_norm = normalizer.normalize_act(act)

    if obs_noise_sigma > 0.0:
        obs_norm = obs_norm + obs_noise_sigma * torch.randn_like(obs_norm)
        obs_for_target = obs_norm * torch.as_tensor(
            normalizer.obs_std, dtype=obs.dtype, device=obs.device
        ) + torch.as_tensor(normalizer.obs_mean, dtype=obs.dtype, device=obs.device)
        target_delta = next_obs - obs_for_target
    else:
        target_delta = next_obs - obs

    target_norm = normalizer.normalize_delta(target_delta)
    pred_norm, _ = model(obs_norm, act_norm, None)
    return F.smooth_l1_loss(pred_norm, target_norm, beta=0.5)


def rollout_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    *,
    warmup_steps: int,
    horizon: int,
    noise_sigma: float = 0.0,
) -> torch.Tensor:
    """Open-loop rollout loss with bounded per-step Huber and growing weights."""
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for rollout loss: "
            f"need at least {needed_states - 1} actions for warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed_states
    if max_start > 0:
        start = int(torch.randint(0, max_start + 1, (), device=states.device).item())
    else:
        start = 0
    sub_states = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]
    preds = open_loop_rollout(
        model,
        sub_states,
        sub_actions,
        normalizer,
        warmup_steps=warmup_steps,
        horizon=horizon,
        noise_sigma=noise_sigma,
    )
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]
    pred_norm = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)
    per_element = F.smooth_l1_loss(pred_norm, target_norm, beta=0.25, reduction="none")
    # Bound a single bad step so a divergent batch doesn't blow up the gradients.
    per_element = torch.clamp(per_element, max=4.0)
    per_step = per_element.mean(dim=(0, 2))
    step_idx = torch.arange(1, int(horizon) + 1, dtype=per_step.dtype, device=per_step.device)
    step_weights = 1.0 + step_idx / float(horizon)  # 1.0 -> 2.0 across the horizon
    return torch.sum(per_step * step_weights) / torch.sum(step_weights)


def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    _sync_model_normalizer(model, normalizer)
    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]
    _initialize_model(model, states, actions)

    # Reflection augmentation: randomly negate ~50% of windows in the batch.
    flip_prob = float(loss_cfg.get("symmetry_flip_probability", 0.5))
    states, actions = _reflect_augment(states, actions, flip_probability=flip_prob)

    obs_noise_sigma = float(loss_cfg.get("obs_noise_sigma", 0.0))
    rollout_noise_sigma = float(loss_cfg.get("rollout_noise_sigma", 0.0))

    one = one_step_delta_loss(
        model, states, actions, normalizer, obs_noise_sigma=obs_noise_sigma
    )

    max_horizon = int(loss_cfg.get("rollout_train_horizon", 50))
    min_horizon = int(loss_cfg.get("rollout_min_horizon", min(5, max_horizon)))
    min_horizon = max(1, min(min_horizon, max_horizon))
    warmup = int(cfg.get("eval", {}).get("warmup_steps", 10))

    # Hard ceiling so we never request more rollout than the batch supports.
    feasible_horizon = max(1, int(states.shape[1]) - warmup - 1)
    max_horizon = max(1, min(max_horizon, feasible_horizon))
    min_horizon = min(min_horizon, max_horizon)

    if min_horizon < max_horizon:
        horizon = int(
            torch.randint(min_horizon, max_horizon + 1, (), device=states.device).item()
        )
    else:
        horizon = max_horizon

    roll = rollout_loss(
        model,
        states,
        actions,
        normalizer,
        warmup_steps=warmup,
        horizon=horizon,
        noise_sigma=rollout_noise_sigma,
    )
    total = (
        float(loss_cfg.get("one_step_weight", 1.0)) * one
        + float(loss_cfg.get("rollout_weight", 1.0)) * roll
    )
    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
        "loss/rollout_horizon": float(horizon),
    }
