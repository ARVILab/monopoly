import torch
import torch.nn as nn
import torch.nn.functional as F
from policies.mlp import MLP
from policies.resnet import ResNet
from utils.distributions import Categorical

import numpy as np

class NNWrapper(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super(NNWrapper, self).__init__()

        # self.base = ResNet(obs_shape, action_shape)
        self.base = MLP(obs_shape, action_shape)

        self.dist_layer = Categorical()


    def forward(self, *args):
         raise NotImplementedError

    def act(self, state, cash, mask):
        value, action_features = self.base(state)
        dist = self.dist_layer.act(action_features, mask=mask, money=cash)

        action = dist.sample()

        action_log_probs = dist.log_probs(action)

        # for i in range(29, 57):
        #     action[0][i] -= state[0][-1].item()

        return value, action, action_log_probs

    def eval_action(self, state, action):
        value, action_features = self.base(state)
        dist = self.dist_layer.act(action_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy

    def get_value(self, state):
        value, _ = self.base(state)
        return value

    def auction_policy(self, max_bid, org_price, state, cash):
        if max_bid >= org_price * 2:
            return True, 0

        if cash >= max_bid * 3:
            bid = max_bid + int(0.1 * org_price)
            return False, bid

        return True, 0

    def jail_policy(self, state, cash, mask):   # need info about amount of card available
        value, action_features = self.base(state)
        dist = self.dist_layer.act(action_features, mask=mask, money=cash)

        action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs
