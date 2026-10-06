# replay memory

git remote -v
# replay memory

"""
Experience Replay Buffer for DQN.

Stores transitions of the form:

    (state, action, reward, next_state, done)

and allows random mini-batch sampling during training.
"""

import random
from collections import deque
from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    """
    Fixed-size replay memory for DQN.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions stored in memory.
        When full, the oldest transition is removed automatically.
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        """
        Store one transition.

        Parameters
        ----------
        state
            Current state.
        action
            Action taken in the current state.
        reward
            Reward received.
        next_state
            State reached after taking the action.
        done
            True if the episode ended, otherwise False.
        """

        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    def sample(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor
    ]:
        """
        Randomly sample a mini-batch from replay memory.

        Returns
        -------
        states
            Shape: (batch_size, 4, 84, 84)

        actions
            Shape: (batch_size,)

        rewards
            Shape: (batch_size,)

        next_states
            Shape: (batch_size, 4, 84, 84)

        dones
            Shape: (batch_size,)
        """

        batch = random.sample(
            self.buffer,
            batch_size
        )

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.from_numpy(
            np.stack(states)
        ).to(device)

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=device
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=device
        )

        next_states = torch.from_numpy(
            np.stack(next_states)
        ).to(device)

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=device
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    def __len__(self):
        """
        Return the current number of stored transitions.
        """
        return len(self.buffer)


if __name__ == "__main__":

    buffer = ReplayBuffer(capacity=100)

    # Add 10 fake Atari transitions
    for i in range(10):

        state = np.random.randint(
            0,
            256,
            size=(4, 84, 84),
            dtype=np.uint8
        )

        next_state = np.random.randint(
            0,
            256,
            size=(4, 84, 84),
            dtype=np.uint8
        )

        action = np.random.randint(0, 4)

        reward = float(
            np.random.choice([-1, 0, 1])
        )

        done = bool(
            np.random.choice([False, True])
        )

        buffer.push(
            state,
            action,
            reward,
            next_state,
            done
        )

    print("Replay buffer size:", len(buffer))

    device = torch.device("cpu")

    states, actions, rewards, next_states, dones = \
        buffer.sample(
            batch_size=4,
            device=device
        )

    print("States shape     :", states.shape)
    print("Actions shape    :", actions.shape)
    print("Rewards shape    :", rewards.shape)
    print("Next states shape:", next_states.shape)
    print("Dones shape      :", dones.shape)

    print("\nActions:", actions)
    print("Rewards:", rewards)
    print("Dones  :", dones)