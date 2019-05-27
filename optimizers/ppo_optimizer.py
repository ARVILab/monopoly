import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from tqdm import tqdm
import config


class PPO(object):
    def __init__(self,
                policy,
                clip_param,
                ppo_epoch,
                mini_batch_size,
                value_loss_coef,
                entropy_coef,
                lr,
                max_grad_norm):

        self.policy = policy

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.mini_batch_size = mini_batch_size
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.epsilon = 1e-8

        self.optimizer = optim.Adam(self.policy.policy.parameters(), lr=lr, eps=self.epsilon)

        self.mse_criterion = nn.MSELoss()
        self.crossentropy_criterion = nn.CrossEntropyLoss()

    def update(self, storage):
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        print('------Samples:', len(storage.rewards))

        for e in tqdm(range(self.ppo_epoch)):
            data_generator = storage.sample(self.mini_batch_size)

            for sample in data_generator:
                states_batch, actions_batch, old_log_probs_batch, returns_batch, adv_targ = sample


                values, actions_pred = self.policy.pred_action(states_batch)

                # actions_batch_one_hot = self.one_hot_encoder(actions_batch, actions_pred.size()[1])

                # action_loss = 0.5 * self.mse_criterion(actions_batch_one_hot, actions_pred)

                action_loss = self.crossentropy_criterion(actions_pred, actions_batch.squeeze(-1).long())

                # values, log_probs, dist_entropy = self.policy.eval_action(states_batch, actions_batch)
                #
                # ratio = torch.exp(log_probs - old_log_probs_batch)
                # surr1 = ratio * adv_targ
                # surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                # action_loss = -torch.min(surr1, surr2).mean()

                value_loss = self.mse_criterion(values, returns_batch)

                self.optimizer.zero_grad()

                # loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                loss = action_loss + value_loss
                loss.backward()

                # nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                # dist_entropy_epoch += dist_entropy.item()

        n_updates = self.ppo_epoch * self.mini_batch_size

        value_loss_epoch /= n_updates
        action_loss_epoch /= n_updates
        dist_entropy_epoch /= n_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def one_hot_encoder(self, actions, size):
        out = torch.zeros((actions.size()[0], 60))
        for i in range(actions.size()[0]):
            indx = int(actions[i].item())
            out[i][indx] = 1.
        return out.to(config.device)

