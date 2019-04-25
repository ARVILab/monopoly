import numpy as np
import torch
import config


class FixedAgent(object):

    def __init__(self, high, low, jail):
        self.high = high
        self.low = low
        self.jail = jail
        self.device = config.device

    def act(self, state, cash, mask, mortgages=None, buyings=None):
        actions = np.zeros(60)
        if cash >= self.high:
            for i in range(1, 29):
                actions[i] = 1
        elif cash <= self.low:
            for i in range(29, 57):
                actions[i] = 1
        else:
            actions[0] = 1

        actions = actions * mask.cpu().numpy()

        action = np.array([actions.argmax()])
        if action == 59:
            action = 0

        action = torch.from_numpy(np.array(action)).float().to(self.device).view(1, -1)
        value = torch.from_numpy(np.array([0])).float().to(self.device).view(1, -1)
        log_prob = torch.from_numpy(np.array([0])).float().to(self.device).view(1, -1)
        return value, action, log_prob

    def auction_policy(self, max_bid, org_price, state, cash):
        if max_bid >= org_price * 2:
            return True, 0

        if cash >= max_bid * 3:
            bid = max_bid + int(0.1 * org_price)
            return False, bid

        return True, 0

    def jail_policy(self, state, cash, mask):   # need info about amount of card available
        actions = np.zeros(60)
        if cash >= self.jail:
            actions[57] = 1
        else:
            actions[0] = 1

        actions = actions * mask.cpu().numpy()

        action = np.array([actions.argmax()])
        if action == 59:
            action = 0

        action = torch.from_numpy(action).float().to(self.device).view(1, -1)
        value = torch.from_numpy(np.array([0])).float().to(self.device).view(1, -1)
        log_prob = torch.from_numpy(np.array([0])).float().to(self.device).view(1, -1)
        return value, action, log_prob

    def get_value(self):
        value = torch.from_numpy(np.array([12.])).float().to(self.device).view(1, -1)
        return value
