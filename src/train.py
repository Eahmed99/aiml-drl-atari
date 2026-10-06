"""Run from the repository root: python -m src.train --config config/smoke.json."""
import argparse
import csv
import importlib.metadata
import time
from pathlib import Path
import numpy as np
import torch
from .agent import optimize_model
from .dqn import DQN
from .environment import make_env, as_state, life_count, choose_action
from .replay_buffer import ReplayBuffer
from .utils import ROOT, load_config, seed_everything, get_device, epsilon_at, save_json


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/breakout.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results/run")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run(load_config(args.config), args.out, args.device)


if __name__ == "__main__":
    main()
