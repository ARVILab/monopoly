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

    # args.model = 1

    config.train_on_fixed = True
    if args.model == 'init':
        config.train_on_fixed = True
        policy = NNWrapper('actor_critic', config.state_space, config.action_space, config.train_on_fixed)
        policy.to(config.device)
    else:
        policy = torch.load('./supervised_models/model-60.pt', map_location=lambda storage, loc: storage)

    storage_class = StoragePPO
    trainer = Trainer(policy, storage_class=storage_class, n_episodes=5000, n_games_per_eps=2, n_rounds=5, n_eval_games=20, verbose_eval=20,
                      checkpoint_step=1, reset_files=True, train_on_fixed=config.train_on_fixed)
    start = datetime.datetime.now()
    trainer.run()
    end = datetime.datetime.now()
    diff = end - start
    print('Training took {} min'.format(np.round(diff.total_seconds() / 60, 3)))


if __name__ == '__main__':
    main()
