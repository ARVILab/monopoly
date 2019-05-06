import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, obs_shape, output_shape, hidden_size=128):
        super(DQN, self).__init__()

        self.n_inputs = obs_shape
        self._hidden_size = hidden_size

        self.model = nn.Sequential(
            nn.Linear(self.n_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_shape)
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    def forward(self, x):
        return self.model(x)
