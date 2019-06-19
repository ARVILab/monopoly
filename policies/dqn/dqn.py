import torch
import torch.nn as nn

import numpy as np

class DQN(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=64):
        super(DQN, self).__init__()

        self.num_outputs = num_outputs

        self.base = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_outputs)
        )

        self.train()

        self.epsilon = 0

    def forward(self, x):
        return self.base(x)

    def act(self, state, mask):
        if np.random.rand() > self.epsilon:
            q_values = self.base(state)
            q_values = q_values * mask
            q_values = torch.where(q_values == 0, q_values, torch.exp(q_values))
            action = q_values.cpu().argmax(1).unsqueeze(0)
        else:
            actions = np.random.rand(self.num_outputs)
            actions = actions * mask.cpu().numpy()
            action = torch.LongTensor([[actions.argmax()]])     # apply mask if needed
        return action

    def update_epsilon(self, epsilon):
        self.epsilon = epsilon
