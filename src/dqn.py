"""Q network for four stacked Atari frames."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, num_actions: int):
        super().__init__()

        # Input: four grayscale 84x84 frames.
        self.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=16,
            kernel_size=8,
            stride=4
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=2
        )

        # The convolutions leave 32 feature maps of size 9x9.
        self.fc1 = nn.Linear(
            32 * 9 * 9,
            256
        )

        # One Q-value for each possible action
        self.fc2 = nn.Linear(
            256,
            num_actions
        )

    def forward(self, x):
        """Return action Q-values for a batch shaped (N, 4, 84, 84)."""

        # Normalize the uint8 pixels stored in replay.
        x = x.float() / 255.0

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten everything except batch dimension
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))

        # No final ReLU: Q-values can be negative.
        q_values = self.fc2(x)

        return q_values

if __name__ == "__main__":

    num_actions = 4

    model = DQN(num_actions)

    # Check the output shape with a batch of random frames.
    dummy_state = torch.randint(
        0,
        256,
        (32, 4, 84, 84),
        dtype=torch.uint8
    )

    q_values = model(dummy_state)

    print(model)
    print("Input shape :", dummy_state.shape)
    print("Output shape:", q_values.shape)
