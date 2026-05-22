"""Render a checkpoint rollout as a compact ground-truth vs prediction video."""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from .checkpoint import load_checkpoint
from .config import load_config
from .dataset import load_metadata, load_split, validate_split_against_metadata
from .horizon import resolve_eval_horizon
from .normalizer import Normalizer
from .official_rollout import official_open_loop_rollout


def _trajectory_from_checkpoint(
    checkpoint_dir: str | Path,
    dataset_dir: str | Path,
    split: str,
    *,
    window_index: int,
    warmup_steps: int | None,
    horizon: int,
    eval_config: str | Path | None,
    device: str | torch.device | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    torch_device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, payload = load_checkpoint(checkpoint_dir, device=torch_device)
    data = load_split(dataset_dir, split)
    metadata = load_metadata(dataset_dir)
    validate_split_against_metadata(data, metadata, split)
    if not 0 <= int(window_index) < int(data["states"].shape[0]):
        raise IndexError(f"window_index={window_index} outside split with {data['states'].shape[0]} windows.")

    eval_cfg = load_config(eval_config) if eval_config is not None else None
    eval_settings = dict(payload["config"].get("eval", {}))
    if eval_cfg is not None:
        eval_settings.update(eval_cfg.get("eval", eval_cfg))
    merged_cfg = {**payload["config"], "eval": eval_settings}
    states_np = data["states"][int(window_index) : int(window_index) + 1]
    actions_np = data["actions"][int(window_index) : int(window_index) + 1]
    warmup, resolved_horizon = resolve_eval_horizon(
        states_shape=tuple(states_np.shape),
        actions_shape=tuple(actions_np.shape),
        cfg=merged_cfg,
        warmup_override=warmup_steps,
        horizon_override=int(horizon),
    )
    states = torch.as_tensor(states_np, dtype=torch.float32, device=torch_device)
    actions = torch.as_tensor(actions_np, dtype=torch.float32, device=torch_device)
    normalizer = Normalizer.from_dict(payload["normalizer"])
    with torch.no_grad():
        preds = official_open_loop_rollout(model, states, actions, normalizer, warmup_steps=warmup, horizon=resolved_horizon)
    truth = states_np[0, warmup : warmup + resolved_horizon + 1]
    pred = np.concatenate([states_np[0, warmup : warmup + 1], preds.detach().cpu().numpy()[0]], axis=0)
    return truth, pred, warmup


def _cart_pole_points(state: np.ndarray, pole_length: float) -> tuple[float, float, float]:
    cart_x = float(state[0])
    theta = float(state[1])
    pole_x = cart_x + pole_length * np.sin(theta)
    pole_y = pole_length * np.cos(theta)
    return cart_x, pole_x, pole_y


def render_rollout_video(
    checkpoint_dir: str | Path,
    dataset_dir: str | Path,
    split: str,
    output_path: str | Path,
    *,
    window_index: int = 0,
    warmup_steps: int | None = None,
    horizon: int = 200,
    eval_config: str | Path | None = None,
    fps: int = 30,
    device: str | torch.device | None = None,
) -> dict[str, str | int]:
    truth, pred, warmup = _trajectory_from_checkpoint(
        checkpoint_dir,
        dataset_dir,
        split,
        window_index=window_index,
        warmup_steps=warmup_steps,
        horizon=horizon,
        eval_config=eval_config,
        device=device,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pole_length = 1.0
    cart_w, cart_h = 0.18, 0.10
    all_x = np.concatenate([truth[:, 0], pred[:, 0]])
    x_min = min(-1.2, float(all_x.min()) - 0.5)
    x_max = max(1.2, float(all_x.max()) + 0.5)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.2, 1.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("cart position")
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.25)
    ax.axhline(0, color="#333333", linewidth=1.2)
    title = ax.set_title("")

    truth_cart = plt.Rectangle((0, 0), cart_w, cart_h, color="#222222", alpha=0.9)
    pred_cart = plt.Rectangle((0, 0), cart_w, cart_h, color="#1f77b4", alpha=0.35)
    ax.add_patch(truth_cart)
    ax.add_patch(pred_cart)
    (truth_pole,) = ax.plot([], [], color="#222222", linewidth=3.0, label="ground truth")
    (pred_pole,) = ax.plot([], [], color="#1f77b4", linewidth=2.5, linestyle="--", label="model prediction")
    (truth_tip,) = ax.plot([], [], "o", color="#222222", markersize=5)
    (pred_tip,) = ax.plot([], [], "o", color="#1f77b4", markersize=5)
    ax.legend(loc="upper right")

    def update(frame: int):
        tx, tpx, tpy = _cart_pole_points(truth[frame], pole_length)
        px, ppx, ppy = _cart_pole_points(pred[frame], pole_length)
        truth_cart.set_xy((tx - cart_w / 2.0, -cart_h / 2.0))
        pred_cart.set_xy((px - cart_w / 2.0, -cart_h / 2.0))
        truth_pole.set_data([tx, tpx], [0, tpy])
        pred_pole.set_data([px, ppx], [0, ppy])
        truth_tip.set_data([tpx], [tpy])
        pred_tip.set_data([ppx], [ppy])
        title.set_text(f"{split} window {window_index} | warmup {warmup} | prediction step {frame}")
        return truth_cart, pred_cart, truth_pole, pred_pole, truth_tip, pred_tip, title

    ani = animation.FuncAnimation(fig, update, frames=len(truth), interval=1000 / int(fps), blit=True)
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=int(fps))
    else:
        writer = animation.FFMpegWriter(fps=int(fps), bitrate=1800)
    ani.save(output_path, writer=writer)
    plt.close(fig)
    return {
        "video": str(output_path),
        "split": split,
        "window_index": int(window_index),
        "warmup_steps": int(warmup),
        "frames": int(len(truth)),
        "fps": int(fps),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="artifacts/student/best_checkpoint")
    parser.add_argument("--dataset-dir", default="data/dev")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="artifacts/student/rollout_video.mp4")
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--eval-config", default="configs/official_eval.yaml")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            render_rollout_video(
                args.checkpoint_dir,
                args.dataset_dir,
                args.split,
                args.output,
                window_index=args.window_index,
                warmup_steps=args.warmup,
                horizon=args.horizon,
                eval_config=args.eval_config,
                fps=args.fps,
                device=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
