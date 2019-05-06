from trainer import Trainer
from nn_wrapper import NNWrapper
from arena import Arena
from policies.random import RandomAgent
from policies.fixed import FixedAgent
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
    print('device', config.device)

    if args.model == 'init':
        policy = NNWrapper(config.state_space, config.action_space)
        policy.call_counter = 0
        policy.to(config.device)
    else:
        policy = torch.load('./models/model.pt')
        
    policy.call_counter = 0

    trainer = Trainer(policy, n_episodes=5000, n_games_per_eps=1, n_rounds=5000, n_eval_games=10, verbose_eval=20,
                      checkpoint_step=5, reset_files=True)
    start = datetime.datetime.now()
    trainer.run()
    end = datetime.datetime.now()
    diff = end - start
    print('Training took {} min'.format(np.round(diff.total_seconds() / 60, 3)))


if __name__ == '__main__':
    main()
