"""Capture raw frame, input stack and network evidence before training."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from .environment import make_env, as_state
from .dqn import DQN
from .utils import ROOT, load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config/smoke.json")
    p.add_argument("--out", type=Path, default=ROOT / "screenshots")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    env = make_env(config, "rgb_array")
    try:
        obs, _ = env.reset(seed=config["seed"])
        for _ in range(20):
            obs, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                obs, _ = env.reset()
        state = as_state(obs)
        plt.imsave(args.out / "environment.png", env.render())
        fig, axes = plt.subplots(1, 4, figsize=(10, 3))
        for i, ax in enumerate(axes):
            ax.imshow(state[i], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"Stack frame {i + 1}")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(args.out / "frame_stack.png")
        plt.close(fig)
        model = DQN(env.action_space.n)
        with torch.no_grad():
            q = model(torch.as_tensor(state).unsqueeze(0))
        (args.out / "network_summary.txt").write_text(
            f"{model}\nInput: {state.shape} {state.dtype}\nQ shape: {tuple(q.shape)}\n"
            f"Untrained Q values: {q.tolist()}\nActions: {env.unwrapped.get_action_meanings()}",
            encoding="utf-8")
        print("Evidence saved to", args.out)
    finally:
        env.close()


if __name__ == "__main__":
    main()
