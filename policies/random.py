import numpy as np

class RandomAgent(object):

    def __init__(self):
        pass

    def act(self, obs, cash):
        action = np.random.rand(60)
        for i in range(29, 57):
            action[i] -= obs[-1]
        return action

    def auction_policy(self, max_bid, org_price, obs, cash):
        resign = np.random.rand(1)[0]
        if resign > 0.5:
            return True, 0

        bid = max_bid + int(np.random.rand(1)[0] * 100) + 0.
        return False, bid

    def jail_policy(self, state, cash):   # need info about amount of card available
        action = np.random.rand(60)
        return action
