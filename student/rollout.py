"""Student open-loop rollout implementation.

The default behaviour matches the locked official rollout exactly so that
training and evaluation share the same trajectory shape.  An optional
``noise_sigma`` argument adds small Gaussian perturbations (scaled by the
training-set observation std) to each predicted state before feeding it back
into the model.  This is *training-only* scheduled sampling: it teaches the
model to recover from realistic accumulated errors and is a major contributor
to long-horizon stability.  Callers from evaluation paths should leave
``noise_sigma=0.0``.
"""

from __future__ import annotations

import torch

from wm_hw.model_utils import predict_next


def open_loop_rollout(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
    *,
    noise_sigma: float = 0.0,
    noise_generator: torch.Generator | None = None,
):
    """Roll out `horizon` steps after a ground-truth warmup.

    Future ground-truth states after `warmup_steps` must not be read.
    """
    batch_size = states.shape[0]
    hidden = model.initial_hidden(batch_size, states.device)
    for t in range(int(warmup_steps)):
        _, hidden = predict_next(model, states[:, t], actions[:, t], hidden, normalizer)
    cur = states[:, int(warmup_steps)]
    preds = []
    sigma = float(noise_sigma)
    if sigma > 0.0:
        obs_std = torch.as_tensor(
            normalizer.obs_std, dtype=cur.dtype, device=cur.device
        ).clamp_min(1e-6)
    for h in range(int(horizon)):
        cur, hidden = predict_next(
            model, cur, actions[:, int(warmup_steps) + h], hidden, normalizer
        )
        preds.append(cur)
        if sigma > 0.0 and h < int(horizon) - 1:
            if noise_generator is None:
                noise = torch.randn_like(cur)
            else:
                noise = torch.randn(cur.shape, generator=noise_generator, device=cur.device, dtype=cur.dtype)
            cur = cur + sigma * obs_std * noise
    return torch.stack(preds, dim=1)
