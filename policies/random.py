import numpy as np
import torch
import config


class RandomAgent(object):

    def __init__(self):
        self.device = config.device

    def act(self, state, cash, mask, survive=False, money_owned=0, mortgages=None, buyings=None):
        actions = np.random.rand(60)
        for i in range(29, 57):
            actions[i] -= state[0][-1].item()

        actions = actions * mask.cpu().numpy()

        action = np.array([actions.argmax()])

        value = torch.FloatTensor([0.]).unsqueeze(0).to(self.device)
        action = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        log_prob = torch.FloatTensor([0.]).unsqueeze(0).to(self.device)
        # return value, action, log_prob
        return action

    def auction_policy(self, max_bid, org_price, obs, cash):
        resign = np.random.rand(1)[0]
        if resign > 0.5:
            return True, 0

        bid = max_bid + int(np.random.rand(1)[0] * 100) + 0.
        return False, bid

    def jail_policy(self, state, cash, mask):   # need info about amount of card available
        actions = np.random.rand(60)
        actions = actions * mask.cpu().numpy()

        action = np.array([actions.argmax()])

        value = torch.FloatTensor([0.]).unsqueeze(0).to(self.device)
        action = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        log_prob = torch.FloatTensor([0.]).unsqueeze(0).to(self.device)
        # return value, action, log_prob
        return action
