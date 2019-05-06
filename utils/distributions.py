import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from utils.weights_init import AddBias, init, init_normc_

FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)


class Categorical(nn.Module):
    def __init__(self):
        super(Categorical, self).__init__()

    def act(self, input, mask=None, money=0, decay=0, use_decay=False, state=None, mortgages=None, buyings=None):
        with torch.no_grad():
            x = F.softmax(input, dim=1)
            if use_decay:
                x[0][0] = x[0][0].item() - x[0][0].item() * decay

                if state is not None:
                    for i in range(1, 29):
                        x[0][i] = np.clip(x[0][i].item() + state[0][-1].item() * decay, .0001, .999)
                    for i in range(29, 57):
                        x[0][i] = np.clip(x[0][i].item() - state[0][-1].item() * decay, .0001, .999)

            if mask is not None:
                x = x * mask

            if mortgages:
                for elem in mortgages:
                    x[0][elem - 28] = 0.
            if buyings:
                for elem in buyings:
                    x[0][elem + 28] = 0.

        return FixedCategorical(probs=x)

    def forward(self, *args):
        raise NotImplementedError
