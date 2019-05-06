import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

import datetime


class PPO():
    def __init__(self,
                policy,
                clip_param,
                ppo_epoch,
                n_mini_batch,
                value_loss_coef,
                entropy_coef,
                lr,
                epsilon,
                max_grad_norm):

        self.policy = policy

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.n_mini_batch = n_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(policy.base.parameters(), lr=lr, eps=epsilon)
        # torch.optim.Adam(filter(lambda p: p.requires_grad, policy.parameters()), lr)

    def update(self, storage):
        print('---------PPO updates')
        advantages = storage.returns[:-1] - storage.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for _ in tqdm(range(self.ppo_epoch)):
            data_generator = storage.feed_forward_generator(advantages, self.n_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, \
                      masks_batch, old_action_log_probs_batch, adv_targ = sample


                dist, value = model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

                action_loss  = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * F.mse_loss(return_batch, values)

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                optimizer.zero_grad()
                loss.backward()

                # nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        n_updates = self.ppo_epoch * self.n_mini_batch

        value_loss_epoch /= n_updates
        action_loss_epoch /= n_updates
        dist_entropy_epoch /= n_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
