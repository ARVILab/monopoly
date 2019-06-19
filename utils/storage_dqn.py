import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import config


class StorageDQN(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.next_states = []
        self.device = config.device

    def push(self, state, action, reward, next_state, done):
        mask = torch.FloatTensor([done]).to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        state = state.to(self.device)
        next_state = next_state.to(self.device)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.masks.append(mask)
        self.next_states.append(next_state)

    def compute(self):
        self.states = torch.cat(self.states)
        self.actions = torch.cat(self.actions)
        self.rewards = torch.cat(self.rewards)
        self.masks = torch.cat(self.masks)
        self.next_states = torch.cat(self.next_states)

    def sample(self, mini_batch_size):
        batch_size = self.states.size(0)
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            yield self.states[indices, :], self.actions[indices, :], self.rewards[indices, :], \
                  self.next_states[indices, :], self.masks[indices, :]

    def get_mean_reward(self):
        reward = 0
        for i in range(len(self.rewards)):
            reward += self.rewards[i].item()
        return reward / len(self.rewards)
