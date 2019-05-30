import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from tqdm import tqdm
import config


class SupervisedLearning(object):
    def __init__(self,
                 policy,
                 mini_batch_size,
                 n_epochs,
                 value_loss_coef,
                 lr):

        self.policy = policy

        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.value_loss_coef = value_loss_coef

        self.epsilon = 1e-8

        self.optimizer = optim.Adam(self.policy.policy.parameters(), lr=lr, eps=self.epsilon)

        self.mse_criterion = nn.MSELoss()
        self.crossentropy_criterion = nn.CrossEntropyLoss()

    def update(self, storage):
        value_loss_epoch = 0
        action_loss_epoch = 0
        print('------Samples:', len(storage.rewards))

        for e in tqdm(range(self.n_epochs)):
            data_generator = storage.sample(self.mini_batch_size)

            for sample in data_generator:
                states_batch, actions_batch, old_log_probs_batch, returns_batch, adv_targ = sample

                values, actions_pred = self.policy.pred_action(states_batch)
                action_loss = self.crossentropy_criterion(actions_pred, actions_batch.squeeze(-1).long())

                value_loss = self.mse_criterion(values, returns_batch)

                self.optimizer.zero_grad()
                loss = action_loss + value_loss * self.value_loss_coef
                loss.backward()

                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()

        n_updates = self.n_epochs * self.mini_batch_size

        value_loss_epoch /= n_updates
        action_loss_epoch /= n_updates

        return value_loss_epoch, action_loss_epoch, []
