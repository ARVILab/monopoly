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
    print('device', config.device)

    policy = NNWrapper(config.state_space, config.action_space)
    policy.to(config.device)

    trainer = Trainer(policy, n_episodes=100, n_games_per_eps=2, n_rounds=2000, n_eval_games=50, verbose_eval=5, checkpoint_step=5, reset_files=True)
    start = datetime.datetime.now()
    trainer.run()
    end = datetime.datetime.now()
    diff = end - start
    print('Training took {} min'.format(np.round(diff.total_seconds() / 60, 3)))


    # model_eps = list(range(0, 500, 20))
    #
    # models = ['./models/model-{}.pt'.format(eps)for eps in model_eps][:10]
    # arena = Arena(n_games=50)
    # times = []

    # with open('winrates_all_models.csv', 'w') as file:
    #     file.write('model,winrate\n')

    # for model in models:
    #     print('Model:', model)
    #     policy = torch.load(model)
    #     policy.eval()
    #     start = datetime.datetime.now()
    #     winrate = arena.fight(agent=policy, opponent=FixedAgent(high=350, low=150, jail=100))
    #     end = datetime.datetime.now()
        # with open('winrates_all_models.csv', 'a') as file:
        #     file.write('{},{}\n'.format(model, winrate))
        # diff = end - start
        # times.append(diff.total_seconds())

    # print('Avg time for 50 games:', np.average(times))

    # print('ARENA')
    # arena = Arena(n_games=1, verbose=1, n_rounds=100)
    # # policy = torch.load('./models/model-5.pt', map_location=lambda storage, loc: storage)
    # policy = torch.load('./models/model-5.pt')
    # # policy = NNWrapper(config.state_space, config.action_space)
    # # policy.to(config.device)
    #
    # policy.eval()
    # winrate = arena.fight(agent=policy, opponent=FixedAgent(high=350, low=150, jail=100))
    # # winrate = arena.fight(agent=policy, opponent=RandomAgent())

    # print('ARENA')
    # arena = Arena(n_games=1, verbose=1, n_rounds=2)
    # arena.fight(agent=FixedAgent(high=350, low=150, jail=100), opponent=FixedAgent(high=350, low=150, jail=100))

if __name__ == '__main__':
    main()