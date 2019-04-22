import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.weights_init import init, init_normc_

class MLP(nn.Module):
    def __init__(self, obs_shape, hidden_size=128):
        super(MLP, self).__init__()

        self.n_inputs = obs_shape
        self._hidden_size = hidden_size

        init_ = lambda m: init(m,
            init_normc_,
            lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(self.n_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(self.n_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, 1))
        )

        # self.train()

    @property
    def output_size(self):
        return self._hidden_size

    def forward(self, x):
        value = self.critic(x)
        action_features = self.actor(x)
        return value, action_features
