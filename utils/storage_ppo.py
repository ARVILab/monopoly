import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import config


class StoragePPO(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []
        self.returns = []
        self.advantages = []

        self.device = config.device

        self.gamma = 0.99     # discount
        self.tau = 0.95       # gae_coef

    def push(self, state, action, log_prob, value, reward, done):
        mask = torch.FloatTensor([done]).to(self.device)
        action = action.to(self.device)
        value = value.to(self.device)
        log_prob = log_prob.to(self.device)
        reward = reward.to(self.device)
        state = state.to(self.device)

        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.masks.append(mask)

    def compute(self, next_value):
        self.values.append(next_value)
        self.masks.append(self.masks[-1].clone())

        gae = 0
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * self.values[step + 1] * self.masks[step + 1] - self.values[step]
            gae = delta + self.gamma * self.tau * self.masks[step + 1] * gae
            self.returns.insert(0, gae + self.values[step])

        del self.values[-1]
        del self.masks[-1]
        self.states = torch.cat(self.states)
        self.actions = torch.cat(self.actions)
        self.log_probs = torch.cat(self.log_probs)
        self.values = torch.cat(self.values)
        self.returns = torch.cat(self.returns)
        self.rewards = torch.cat(self.rewards)
        self.masks = torch.cat(self.masks)
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-5)

    def sample(self, mini_batch_size):
        batch_size = self.states.size(0)
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            yield self.states[indices, :], self.actions[indices, :], self.log_probs[indices, :], \
                  self.returns[indices, :], self.advantages[indices, :]

    def get_mean_reward(self):
        reward = 0
        for i in range(len(self.rewards)):
            reward += self.rewards[i].item()
        return reward / len(self.rewards)

    def show(self):
        for i in range(self.states.shape[0]):
            print('------------------')
            print('STATE', self.states[i])
            print('ACTION', self.actions[i])
            print('LOG PROB', self.log_probs[i])
            print('REWARD', self.rewards[i])
            print('VALUE', self.values[i])
            print('MASK', self.masks[i])
            print('RETURNS', self.returns[i])
            print('------------------')
