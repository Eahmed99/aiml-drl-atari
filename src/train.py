"""Compact Atari DQN training: preprocessing, learning, plots and demos in one file."""
import argparse
import csv
import importlib.metadata
import json
import random
import time
from pathlib import Path
import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .dqn import DQN
from .replay_buffer import ReplayBuffer

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = {'environment': 'ALE/Breakout-v5', 'seed': 42, 'total_steps': 100000, 'gamma': 0.99, 'learning_rate': 0.00025, 'batch_size': 32, 'replay_capacity': 5000, 'learning_starts': 1000, 'train_frequency': 4, 'epsilon_start': 1.0, 'epsilon_end': 0.1, 'epsilon_decay_steps': 80000, 'sticky_actions': 0.0, 'fire_on_reset': True, 'max_raw_episode_steps': 108000, 'log_every': 1000, 'checkpoint_every': 10000, 'torch_threads': 2}
SMOKE_CONFIG = {'environment': 'ALE/Breakout-v5', 'seed': 42, 'total_steps': 2000, 'gamma': 0.99, 'learning_rate': 0.00025, 'batch_size': 32, 'replay_capacity': 5000, 'learning_starts': 500, 'train_frequency': 4, 'epsilon_start': 1.0, 'epsilon_end': 0.1, 'epsilon_decay_steps': 1500, 'sticky_actions': 0.0, 'fire_on_reset': True, 'max_raw_episode_steps': 10800, 'log_every': 250, 'checkpoint_every': 1000, 'torch_threads': 2}

def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    for key in ("total_steps", "batch_size", "replay_capacity", "train_frequency",
                "epsilon_decay_steps", "max_raw_episode_steps", "log_every",
                "checkpoint_every", "torch_threads"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not config["batch_size"] <= config["learning_starts"] <= config["replay_capacity"]:
        raise ValueError("Require batch_size <= learning_starts <= replay_capacity")
    if not 0 <= config["epsilon_end"] <= config["epsilon_start"] <= 1:
        raise ValueError("Invalid epsilon range")
    if not 0 <= config["gamma"] <= 1 or config["learning_rate"] <= 0:
        raise ValueError("Invalid gamma or learning rate")
    return config


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name="auto"):
    import torch
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def epsilon_at(step, config):
    fraction = min(step / config["epsilon_decay_steps"], 1.0)
    return config["epsilon_start"] + fraction * (
        config["epsilon_end"] - config["epsilon_start"])


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


class FireOnReset(gym.Wrapper):
    """Launch on reset. Later life losses force FIRE through choose_action().

    Reset launch rewards are not replay transitions and are reported separately.
    This game-specific helper is a documented educational deviation.
    """
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        fire = self.unwrapped.get_action_meanings().index("FIRE")
        obs, reward, terminated, truncated, info = self.env.step(fire)
        if terminated or truncated:
            obs, info = self.env.reset()
        info = dict(info)
        info["reset_launch_reward"] = float(reward)
        info["reset_launch_actions"] = 1
        return obs, info


def make_env(config, render_mode=None):
    gym.register_envs(ale_py)
    env = gym.make(config["environment"], frameskip=1,
                   repeat_action_probability=config["sticky_actions"],
                   render_mode=render_mode,
                   max_episode_steps=config["max_raw_episode_steps"])
    env = AtariPreprocessing(env, noop_max=30, frame_skip=4,
                             screen_size=84, grayscale_obs=True,
                             scale_obs=False, terminal_on_life_loss=False)
    if config["fire_on_reset"]:
        env = FireOnReset(env)
    return FrameStackObservation(env, stack_size=4)


def as_state(observation):
    state = np.asarray(observation)
    if state.shape != (4, 84, 84) or state.dtype != np.uint8:
        raise ValueError(f"Unexpected state: {state.shape}, {state.dtype}")
    return state


def life_count(env):
    return env.unwrapped.ale.lives()


def choose_action(model, state, epsilon, rng, action_count, device,
                  force_fire=False, fire_action=None):
    import torch
    if force_fire:
        return fire_action
    if model is None or rng.random() < epsilon:
        return int(rng.integers(action_count))
    with torch.no_grad():
        tensor = torch.as_tensor(state, device=device).unsqueeze(0)
        return int(model(tensor).argmax(dim=1).item())


def td_targets(rewards, next_q, terminated, gamma):
    return rewards + gamma * next_q.max(dim=1).values * (1 - terminated)


def optimize_model(model, optimizer, replay, batch_size, gamma, device):
    states, actions, rewards, next_states, terminated = replay.sample(batch_size, device)
    current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        target = td_targets(rewards, model(next_states), terminated, gamma)
    loss = F.mse_loss(current_q, target)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite TD loss")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def run(config, out, device_name="auto"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "config.json", config)
    save_json(out / "versions.json", {name: importlib.metadata.version(name) for name in
                                     ("torch", "gymnasium", "ale-py", "numpy")})
    seed_everything(config["seed"])
    torch.set_num_threads(config["torch_threads"])
    device = get_device(device_name)
    env = make_env(config)
    rng = np.random.default_rng(config["seed"])
    replay = ReplayBuffer(config["replay_capacity"], config["seed"])
    model = DQN(env.action_space.n).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config["learning_rate"],
                                  alpha=0.95, eps=0.01)
    (out / "network.txt").write_text(str(model), encoding="utf-8")
    print(f"Device: {device}; replay pixels <= {config['replay_capacity'] * 56448 / 1e6:.1f} MB")
    started = time.monotonic()
    updates = episode = episode_steps = 0
    score = clipped_return = 0.0
    last_loss = None
    last_step = 0

    def save_checkpoint(step):
        torch.save({"model": model.state_dict(), "config": config,
                    "action_count": int(env.action_space.n), "step": int(step),
                    "updates": updates}, out / "checkpoint.pt")

    try:
        with (out / "episodes.csv").open("w", newline="") as ef, \
             (out / "losses.csv").open("w", newline="") as lf:
            ew, lw = csv.writer(ef), csv.writer(lf)
            ew.writerow(["episode", "step", "episode_steps", "score", "clipped_return",
                         "epsilon", "terminated", "truncated", "reset_launch_reward"])
            lw.writerow(["step", "update", "loss"])
            obs, info = env.reset(seed=config["seed"])
            state = as_state(obs)
            launch_reward = info.get("reset_launch_reward", 0)
            lives = life_count(env)
            fire_action = env.unwrapped.get_action_meanings().index("FIRE")
            force_fire = False
            for step in range(1, config["total_steps"] + 1):
                epsilon = epsilon_at(step - 1, config)
                action = choose_action(model, state, epsilon, rng, env.action_space.n,
                                       device, force_fire, fire_action)
                obs, reward, terminated, truncated, info = env.step(action)
                next_state = as_state(obs)
                current_lives = life_count(env)
                force_fire = bool(config["fire_on_reset"] and current_lives < lives
                                  and not (terminated or truncated))
                lives = current_lives
                replay.push(state, action, np.sign(reward), next_state, terminated)
                state = next_state
                episode_steps += 1
                score += float(reward)
                clipped_return += float(np.sign(reward))
                if len(replay) >= config["learning_starts"] and step % config["train_frequency"] == 0:
                    last_loss = optimize_model(model, optimizer, replay,
                                               config["batch_size"], config["gamma"], device)
                    updates += 1
                    lw.writerow([step, updates, last_loss])
                last_step = step
                if terminated or truncated:
                    episode += 1
                    ew.writerow([episode, step, episode_steps, score, clipped_return,
                                 epsilon, terminated, truncated, launch_reward])
                    ef.flush()
                    print(f"Episode {episode}: score={score:.1f}, steps={step}, loss={last_loss}")
                    obs, info = env.reset()
                    state = as_state(obs)
                    launch_reward = info.get("reset_launch_reward", 0)
                    lives = life_count(env)
                    force_fire = False
                    episode_steps = 0
                    score = clipped_return = 0.0
                if step % config["log_every"] == 0:
                    lf.flush()
                    print(f"Step {step}: epsilon={epsilon:.3f}, updates={updates}, loss={last_loss}")
                if step % config["checkpoint_every"] == 0:
                    save_checkpoint(step)
            save_checkpoint(last_step)
    finally:
        env.close()
        save_json(out / "summary.json", {
            "completed_steps": last_step, "updates": updates, "completed_episodes": episode,
            "last_loss": last_loss, "seconds": time.monotonic() - started,
            "partial_episode_score": score, "partial_episode_steps": episode_steps,
            "note": "Partial episode excluded from episodes.csv. Steps count agent actions, not raw frames."
        })
    if updates == 0:
        raise RuntimeError("Run ended without any learning updates")
    return out


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot_results(run, evaluation=None):
    run = Path(run)
    evaluation = Path(evaluation) if evaluation else None
    for file, ykey, title in [
        ("episodes.csv", "score", "Training episode score (unclipped)"),
        ("losses.csv", "loss", "Sampled TD loss (not a performance metric)")]:
        if not (run / file).exists():
            continue
        rows = read_rows(run / file)
        if not rows:
            print(f"No rows in {file}; plot skipped")
            continue
        x = np.array([float(r["step"]) for r in rows])
        y = np.array([float(r[ykey]) for r in rows])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, alpha=0.4, label="Raw")
        window = min(20, max(1, len(y) // 2))
        ax.plot(x[window - 1:], np.convolve(y, np.ones(window) / window, mode="valid"),
                label=f"{window}-point moving mean")
        ax.set(xlabel="Agent steps", ylabel=ykey, title=title)
        ax.legend()
        fig.tight_layout()
        name = "training_reward_curve.png" if ykey == "score" else "loss_curve.png"
        fig.savefig(run / name)
        plt.close(fig)
    if evaluation:
        rows = read_rows(evaluation / "evaluation.csv")
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, label in enumerate(("random", "trained")):
            scores = [float(r["score"]) for r in rows if r["policy"] == label]
            ax.scatter([i] * len(scores), scores, alpha=0.6)
            ax.plot([i - 0.15, i + 0.15], [np.mean(scores)] * 2, color="black")
        ax.set(xticks=[0, 1], xticklabels=["Random", "Trained"], ylabel="Unclipped score",
               title="Evaluation episodes; black line = mean")
        fig.tight_layout()
        fig.savefig(evaluation / "comparison.png")
        plt.close(fig)


def environment_demo(config, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(config, "rgb_array")
    try:
        obs, _ = env.reset(seed=config["seed"])
        for _ in range(20):
            obs, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                obs, _ = env.reset()
        state = as_state(obs)
        plt.imsave(out / "environment.png", env.render())
        fig, axes = plt.subplots(1, 4, figsize=(10, 3))
        for i, ax in enumerate(axes):
            ax.imshow(state[i], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"Stack frame {i + 1}")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / "frame_stack.png")
        plt.close(fig)
        model = DQN(env.action_space.n)
        with torch.no_grad():
            q = model(torch.as_tensor(state).unsqueeze(0))
        (out / "network_summary.txt").write_text(
            f"{model}\nInput: {state.shape} {state.dtype}\nQ shape: {tuple(q.shape)}\n"
            f"Untrained Q values: {q.tolist()}\nActions: {env.unwrapped.get_action_meanings()}",
            encoding="utf-8")
        print("Evidence saved to", out)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Short 2,000-step check")
    parser.add_argument("--steps", type=int, help="Override training agent steps")
    parser.add_argument("--config", type=Path, help="Optional external JSON config")
    parser.add_argument("--out", type=Path, default=ROOT / "results/run")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--demo", action="store_true", help="Capture environment/input evidence")
    parser.add_argument("--plot-run", type=Path, help="Plot an existing run instead of training")
    parser.add_argument("--evaluation", type=Path, help="Evaluation folder to include in plots")
    args = parser.parse_args()
    if args.plot_run:
        plot_results(args.plot_run, args.evaluation)
        return
    config = load_config(args.config) if args.config else dict(SMOKE_CONFIG if args.smoke else DEFAULT_CONFIG)
    if args.steps is not None:
        if args.steps <= 0:
            parser.error("--steps must be positive")
        config["total_steps"] = args.steps
    if args.demo:
        environment_demo(config, ROOT / "screenshots")
        return
    run(config, args.out, args.device)
    plot_results(args.out)


if __name__ == "__main__":
    main()
