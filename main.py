from trainer import Trainer
from nn_wrapper import NNWrapper
from utils.storage_ppo import StoragePPO
from utils.storage_dqn import StorageDQN
import config

import torch
import os
import datetime
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='init', help='model to load; to load specific model use model number')
    args = parser.parse_args()

    config.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # config.device = 'cpu'
    print('device', config.device)

    # args.model = 20

    config.train_on_fixed = True
    config.self_play = False

    if args.model == 'init':
        policy = NNWrapper('actor_critic', config.state_space, config.action_space, config.train_on_fixed)
        policy.to(config.device)
    else:
        if config.device.type == 'cpu':
            policy = torch.load('./supervised_models/model-{}.pt'.format(args.model), map_location=lambda storage, loc: storage)
        else:
            policy = torch.load('./supervised_models/model-{}.pt'.format(args.model))

        policy.train_on_fixed = config.train_on_fixed

    storage_class = StoragePPO
    trainer = Trainer(policy, storage_class=storage_class, n_episodes=5000, n_games_per_eps=10, n_rounds=500, n_eval_games=10, verbose_eval=10,
                      checkpoint_step=10, reset_files=True, train_on_fixed=config.train_on_fixed, self_play=config.self_play)
    start = datetime.datetime.now()
    trainer.run()
    end = datetime.datetime.now()
    diff = end - start
    print('Training took {} min'.format(np.round(diff.total_seconds() / 60, 3)))


if __name__ == '__main__':
    main()
