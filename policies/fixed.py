import numpy as np
import torch

class FixedAgent(object):

    def __init__(self, high, low, jail):
        self.high = high
        self.low = low
        self.jail = jail
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def act(self, state, cash):
        action = np.zeros(60)
        if cash >= self.high:
            for i in range(1, 29):
                action[i] = 1
        elif cash <= self.low:
            for i in range(29, 57):
                action[i] = 1
        else:
            action[0] = 1

        action = torch.from_numpy(action).float().to(self.device).view(1, -1)
        value = torch.from_numpy(np.array([0])).float().to(self.device).view(1, -1)
        log_prob = torch.from_numpy(np.array([0])).float().to(self.device).view(1, -1)
        return value, action, log_prob

    def auction_policy(self, max_bid, org_price, obs, cash):
        if max_bid >= org_price * 2:
            return True, 0

        if cash >= max_bid * 3:
            bid = max_bid + int(0.1 * org_price)
            return False, bid

        return True, 0

    def jail_policy(self, state, cash):   # need info about amount of card available
        action = np.zeros(60)
        if cash >= self.jail:
            action[57] = 1
        else:
            action[0] = 1
        action = torch.from_numpy(action).float().to(self.device).view(1, -1)
        value = torch.from_numpy(np.array([0])).float().to(self.device).view(1, -1)
        log_prob = torch.from_numpy(np.array([0])).float().to(self.device).view(1, -1)
        return value, action, log_prob
