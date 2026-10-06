# neural network + possibly action selection logic

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, num_actions: int):
        super().__init__()

        # First convolutional layer
        # Input: 4 x 84 x 84
        # Output: 16 feature maps
        self.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=16,
            kernel_size=8,
            stride=4
        )

        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=2
        )

        # After convolution:
        # 84x84
        #   ↓ Conv1
        # 20x20
        #   ↓ Conv2
        # 9x9
        #
        # therefore:
        # 32 * 9 * 9 = 2592
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
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Atari state with shape:
            (batch_size, 4, 84, 84)

        Returns
        -------
        torch.Tensor
            Q-values for all actions.
        """

        # Atari frames may arrive as uint8 values from 0-255.
        # Convert to floating point and normalize to 0-1.
        x = x.float() / 255.0

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten everything except batch dimension
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))

        # IMPORTANT:
        # No ReLU on the final layer.
        #
        # Q-values may be positive or negative.
        q_values = self.fc2(x)

        return q_values

if __name__ == "__main__":

    num_actions = 4

    model = DQN(num_actions)

    # Fake batch:
    # 32 states
    # 4 stacked frames
    # each frame 84 x 84
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
