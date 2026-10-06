"""Create plots from real experiment logs."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    args = parser.parse_args()
    for file, ykey, title in [
        ("episodes.csv", "score", "Training episode score (unclipped)"),
        ("losses.csv", "loss", "Sampled TD loss (not a performance metric)")]:
        rows = read_rows(args.run / file)
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
        fig.savefig(args.run / name)
        plt.close(fig)
    if args.evaluation:
        rows = read_rows(args.evaluation / "evaluation.csv")
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, label in enumerate(("random", "trained")):
            scores = [float(r["score"]) for r in rows if r["policy"] == label]
            ax.scatter([i] * len(scores), scores, alpha=0.6)
            ax.plot([i - 0.15, i + 0.15], [np.mean(scores)] * 2, color="black")
        ax.set(xticks=[0, 1], xticklabels=["Random", "Trained"], ylabel="Unclipped score",
               title="Evaluation episodes; black line = mean")
        fig.tight_layout()
        fig.savefig(args.evaluation / "comparison.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
