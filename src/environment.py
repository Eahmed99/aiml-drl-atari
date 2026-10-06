"""Shared training/evaluation preprocessing and explicit Breakout FIRE handling."""
import numpy as np
import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


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
