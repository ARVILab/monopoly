import torch
import torch.nn as nn
import torch.nn.functional as F
from policies.mlp import MLP
from utils.distributions import DiagGaussian

class Policy(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super(Policy, self).__init__()

        self.base = MLP(obs_shape)
        self.dist_layer = DiagGaussian(self.base.output_size, action_shape)

    def forward(self, *args):
         raise NotImplementedError

    def act(self, input):
        value, actor_features = self.base(input)
        dist = self.dist_layer(actor_features)

        action = dist.sample()
        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs

    def eval_action(self, input, action):
        value, actor_features = self.base(input)
        dist = self.dist_layer(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy

    def get_value(self, input):
        value, _ = self.base(input)
        return value