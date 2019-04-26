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


def main():
    config.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # config.device = 'cpu'
    print('device', config.device)

    policy = NNWrapper(config.state_space, config.action_space)
    # policy = torch.load('./models/model-5.pt', map_location=lambda storage, loc: storage)
    # policy = torch.load('./models/model.pt')
    policy.to(config.device)

    trainer = Trainer(policy, n_episodes=5000, n_games_per_eps=1, n_rounds=5000, n_eval_games=20, verbose_eval=10,
                      checkpoint_step=10, reset_files=True)
    start = datetime.datetime.now()
    trainer.run()
    end = datetime.datetime.now()
    diff = end - start
    print('Training took {} min'.format(np.round(diff.total_seconds() / 60, 3)))


    # print('ARENA')
    # arena = Arena(n_games=1, verbose=1, n_rounds=2000)
    # policy = torch.load('./models/model-4120.pt', map_location=lambda storage, loc: storage)
    # # policy = torch.load('./models/model-115.pt')
    # policy.use_decay = False
    # # policy = torch.load('./models/model-5.pt')
    # # policy = NNWrapper(config.state_space, config.action_space)
    # # policy.to(config.device)
    # policy.eval()
    # winrate = arena.fight(agent=policy, opponent=FixedAgent(high=350, low=150, jail=100))
    # winrate = arena.fight(agent=policy, opponent=RandomAgent())


if __name__ == '__main__':
    main()
