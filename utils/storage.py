import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class Storage(object):
    def __init__(self, n_steps, obs_shape, action_shape):
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        self.obs = torch.zeros(n_steps + 1, 1, obs_shape)
        self.rewards = torch.zeros(n_steps, 1, 1)
        self.value_preds = torch.zeros(n_steps + 1, 1, 1)
        self.returns = torch.zeros(n_steps + 1, 1, 1)
        self.action_log_probs = torch.zeros(n_steps, 1, 1)

        self.actions = torch.zeros(n_steps, 1, 1)
        self.actions = self.actions.long()

        self.masks = torch.zeros(n_steps + 1, 1, 1)

        self.n_steps = n_steps
        self.step = 0
        self.counter = 0

    def add_init_obs(self, obs, step=0):
        self.obs[step].copy_(obs)
        self.masks[step].copy_(torch.FloatTensor([1.0]))
        self.step += 1
        self.counter += 1

    def truncate(self):
        obs = torch.zeros(self.counter + 1, 1, self.obs_shape)
        rewards = torch.zeros(self.counter, 1, 1)
        value_preds = torch.zeros(self.counter + 1, 1, 1)
        returns = torch.zeros(self.counter + 1, 1, 1)
        action_log_probs = torch.zeros(self.counter, 1, 1)
        actions = torch.zeros(self.counter, 1, 1)
        masks = torch.zeros(self.counter + 1, 1, 1)

        for step in range(self.counter):
            obs[step].copy_(self.obs[step])
            actions[step].copy_(self.actions[step])
            action_log_probs[step].copy_(self.action_log_probs[step])
            value_preds[step].copy_(self.value_preds[step])
            rewards[step].copy_(self.rewards[step])
            masks[step].copy_(self.masks[step])

        self.obs = obs
        self.rewards = rewards
        self.value_preds = value_preds
        self.returns = returns
        self.action_log_probs = action_log_probs
        self.actions = actions
        self.actions = self.actions.long()
        self.masks = masks

        # print('------------STORAGE TRUNCATED------------')
        # print('COUNTER:', self.counter)
        # for i in range(self.obs.size(0) - 1):
        #     print('--------STEP {}--------'.format(i))
        #     print('OBS:', self.obs[i])
        #     print('ACTIONS:', self.actions[i])
        #     print('LOG PROBS:', self.action_log_probs[i])
        #     print('VALUES:', self.value_preds[i])
        #     print('REWARDS:', self.rewards[i])
        #     print('MASKS:', self.masks[i])

    # def obs_equals(self, elem1, elem2):
    #     r = torch.all(torch.eq(elem1, elem2))
    #     return r.item() == 1
    #
    # def reward_equals(self, elem1, elem2):
    #     return elem1.item() == elem2.item()

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks):
        self.obs[self.step].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step].copy_(torch.FloatTensor(masks))

        self.step = (self.step + 1) % self.n_steps

        self.counter += 1

        # self.show_last_step()

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, gamma, tau):
        print('-----Action taken', self.counter)
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
            gae = delta + gamma * tau * self.masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator(self, advantages, n_mini_batch):
        batch_size = self.rewards.size()[0]
        mini_batch_size = batch_size // n_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def show(self):
        print('------------STORAGE------------'.format(self.counter))
        for i in range(self.obs.size(0) - 1):
            print('--------STEP {}--------'.format(i))
            print('OBS:', self.obs[i])
            print('ACTIONS:', self.actions[i])
            print('LOG PROBS:', self.action_log_probs[i])
            print('VALUES:', self.value_preds[i])
            print('REWARDS:', self.rewards[i])
            print('MASKS:', self.masks[i])
        print('OBS:', self.obs[-1])
        print('VALUES:', self.value_preds[-1])
        print('MASKS:', self.masks[-1])

    def show_last_step(self):
        print('------------STORAGE LAST STEP------------')
        # last_step = self.counter - 1 if self.counter > 1 else 0
        last_step = -1
        print('--------STEP {}--------'.format(last_step))
        print('OBS:', self.obs[last_step])
        print('ACTIONS:', self.actions[last_step])
        print('LOG PROBS:', self.action_log_probs[last_step])
        print('VALUES:', self.value_preds[last_step])
        print('REWARDS:', self.rewards[last_step])
        print('MASKS:', self.masks[last_step])
