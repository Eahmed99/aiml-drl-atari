"""Evaluate a checkpoint and random baseline using matching environment seeds."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from .dqn import DQN
from .train import make_env, as_state, life_count, choose_action, get_device, save_json, plot_results


def evaluate_policy(config, model, seeds, epsilon, device, out, label, video_seconds):
    import imageio.v2 as imageio
    rng = np.random.default_rng(seeds[0])
    env = make_env(config, render_mode="rgb_array" if video_seconds else None)
    rows, writer, recorded = [], None, 0
    fps = 15
    try:
        if video_seconds:
            writer = imageio.get_writer(out / f"{label}_gameplay.mp4", fps=fps)
        fire_action = env.unwrapped.get_action_meanings().index("FIRE")
        for seed in seeds:
            obs, info = env.reset(seed=seed)
            state = as_state(obs)
            launch_reward = info.get("reset_launch_reward", 0)
            lives, force_fire, score, steps = life_count(env), False, 0.0, 0
            terminated = truncated = False
            while not (terminated or truncated):
                action = choose_action(model, state, epsilon, rng, env.action_space.n,
                                       device, force_fire, fire_action)
                obs, reward, terminated, truncated, info = env.step(action)
                state = as_state(obs)
                current_lives = life_count(env)
                force_fire = bool(config["fire_on_reset"] and current_lives < lives
                                  and not (terminated or truncated))
                lives = current_lives
                score += float(reward)
                steps += 1
                if writer is not None and recorded < fps * video_seconds:
                    writer.append_data(env.render())
                    recorded += 1
            rows.append({"policy": label, "seed": seed, "score": score, "steps": steps,
                         "epsilon": epsilon, "terminated": terminated, "truncated": truncated,
                         "reset_launch_reward": launch_reward})
            print(label, seed, score)
    finally:
        env.close()
        if writer is not None:
            writer.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--video-seconds", type=int, default=15)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.episodes <= 0 or not 0 <= args.epsilon <= 1 or args.video_seconds < 0:
        parser.error("Invalid episode count, epsilon or video duration")
    args.out.mkdir(parents=True, exist_ok=False)
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = checkpoint["config"]
    torch.set_num_threads(config["torch_threads"])
    model = DQN(checkpoint["action_count"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    seeds = list(range(args.seed, args.seed + args.episodes))
    rows = []
    for label, policy, epsilon in [("random", None, 1.0), ("trained", model, args.epsilon)]:
        rows.extend(evaluate_policy(config, policy, seeds, epsilon, device,
                                    args.out, label, args.video_seconds))
    with (args.out / "evaluation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for label in ("random", "trained"):
        scores = [r["score"] for r in rows if r["policy"] == label]
        summary[label] = {"mean": float(np.mean(scores)),
                          "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
                          "episodes": len(scores), "scores": scores}
    save_json(args.out / "summary.json", summary)
    save_json(args.out / "protocol.json", {"config": config, "seeds": seeds,
              "checkpoint": str(args.checkpoint), "training_step": checkpoint["step"],
              "epsilon": args.epsilon, "score_units": "unclipped reward excluding reset launch",
              "note": "Environment seeds match; action-dependent trajectories differ."})
    plot_results(args.checkpoint.parent, args.out)
    print(summary)


if __name__ == "__main__":
    main()
