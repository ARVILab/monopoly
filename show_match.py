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

from random import randint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=-1, help='model to load; to load specific model use model number')
    parser.add_argument('--opponent', default='fixed', help='opponent to play against')
    args = parser.parse_args()

    config.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device', config.device)

    # args.opponent = 'random'
    # args.model = 620

    config.train_on_fixed = True
    if args.model == -1 and len(os.listdir('models/')) != 0:
        models = list(filter(lambda name: 'model' in name, os.listdir('./models/')))
        model_number = sorted([int(model_name.split('-')[1].split('.')[0]) for model_name in models])[-1]
        model_name = 'model-{}.pt'.format(model_number)
        print('Loading model:', model_name)
        if config.device.type == 'cpu':
            policy = torch.load(os.path.join('./models', model_name), map_location=lambda storage, loc: storage)
            policy.fixed_agent.device = config.device
        else:
            policy = torch.load(os.path.join('./models', model_name))

        policy.train_on_fixed = False
    elif args.model == 'init' or len(os.listdir('models/')) == 0:
        policy = NNWrapper('actor_critic', config.state_space, config.action_space, config.train_on_fixed)
        policy.to(config.device)
    else:
        model_name = 'model-{}.pt'.format(np.abs(args.model))
        print('Loading model:', model_name)
        if config.device.type == 'cpu':
            policy = torch.load(os.path.join('./models', model_name), map_location=lambda storage, loc: storage)
            policy.fixed_agent.device = config.device
        else:
            policy = torch.load(os.path.join('./models', model_name))
        policy.train_on_fixed = False

    if args.opponent == 'random':
        opponent = RandomAgent()
    else:
        opponent = FixedAgent(high=350, low=150, jail=100)
    policy.eval()

    print('SHOW MATCH')
    arena = Arena(n_games=1, verbose=1, n_rounds=300)

    start = datetime.datetime.now()

    winrate = arena.fight(agent=policy, opponent=opponent, log_rewards=True)

    end = datetime.datetime.now()
    diff = end - start
    print('Took {} sec'.format(np.round(diff.total_seconds(), 3)))

if __name__ == '__main__':
    main()
