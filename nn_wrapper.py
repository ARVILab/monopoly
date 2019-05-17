import torch.nn as nn

from policies.actor_critic.actor_critic import ActorCritic
from policies.dqn.dqn import DQN


class NNWrapper(nn.Module):
    def __init__(self, policy_name, obs_shape, action_shape):
        super(NNWrapper, self).__init__()

        if policy_name == 'actor_critic':
            self.policy = ActorCritic(obs_shape, action_shape, nn='mlp')
        elif policy_name == 'dqn':
            self.policy = DQN(obs_shape, action_shape, nn='mlp')

    def forward(self, *args):
         raise NotImplementedError

    def act(self, state, cash, mask, mortgages=None, buyings=None):
        value, action, log_prob = self.policy.act(state, mask=mask, mortgages=mortgages, buyings=buyings)
        return value, action, log_prob

    def eval_action(self, state, action):
        value, log_prob, entropy = self.policy.eval_action(state, action)
        return value, log_prob, entropy

    def get_value(self, state):
        value, _, _ = self.policy.act(state)
        return value

    def auction_policy(self, max_bid, org_price, state, cash):
        if max_bid >= org_price * 2:
            return True, 0

        if cash >= max_bid * 3:
            bid = max_bid + int(0.1 * org_price)
            return False, bid

        return True, 0

    def jail_policy(self, state, cash, mask):   # need info about amount of card available
        value, action, log_prob = self.policy.act(state, mask=mask)
        return value, action, log_prob
