"""Experience replay for DQN, extending the group's deque-based implementation.

Stores (state, action, reward, next_state, terminated) and samples uniform batches.
Only genuine termination disables bootstrapping; time-limit truncation does not.
"""
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, seed=0):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, terminated):
        state, next_state = np.asarray(state), np.asarray(next_state)
        for frame in (state, next_state):
            if frame.shape != (4, 84, 84) or frame.dtype != np.uint8:
                raise ValueError("States must be uint8 arrays with shape (4, 84, 84)")
        transition = (state.copy(), int(action), float(reward),
                      next_state.copy(), bool(terminated))
        self.buffer.append(transition)

    def sample_arrays(self, batch_size):
        if not 0 < batch_size <= len(self):
            raise ValueError("batch_size must be between 1 and replay length")
        indices = self.rng.choice(len(self), batch_size, replace=False)
        # A deque is not accepted by random.sample on modern Python. A snapshot
        # also avoids repeated O(n) deque indexing when gathering a batch.
        population = list(self.buffer)
        states, actions, rewards, next_states, terminals = zip(
            *(population[int(i)] for i in indices))
        return (np.stack(states), np.asarray(actions, dtype=np.int64),
                np.asarray(rewards, dtype=np.float32), np.stack(next_states),
                np.asarray(terminals, dtype=np.float32))

    def sample(self, batch_size, device="cpu"):
        import torch
        return tuple(torch.as_tensor(x, device=device)
                     for x in self.sample_arrays(batch_size))

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    replay = ReplayBuffer(10)
    for i in range(12):
        frame = np.full((4, 84, 84), i, dtype=np.uint8)
        replay.push(frame, i % 4, 1, frame, i == 11)
    print("Replay length:", len(replay))
    for x in replay.sample_arrays(4):
        print(x.shape, x.dtype)
