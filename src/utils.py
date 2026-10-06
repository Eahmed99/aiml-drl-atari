import json
import random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


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
