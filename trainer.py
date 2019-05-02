from random import shuffle, randint
import sys
import logging
import os
import numpy as np

import torch

from policies.fixed import FixedAgent
from policies.random import RandomAgent
from arena import Arena
from optimizers.ppo import PPO
from monopoly.player import Player
import config
from monopoly.game import Game
from utils.storage import Storage

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, policy, n_episodes=100, n_games_per_eps=10, n_rounds=200, n_eval_games=50, verbose_eval=50, checkpoint_step=10, reset_files=True):
        self.policy = policy
        self.n_games = n_games_per_eps
        self.n_rounds = n_rounds
        self.verbose_eval = verbose_eval
        self.n_eval_games = n_eval_games
        self.checkpoint_step = checkpoint_step
        self.device = config.device

        self.episodes = n_episodes
        self.learning_rate = 1e-3
        self.clip_param = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.alpha = 0.99
        self.max_grad_norm = 0.5
        self.discount = 0.99
        self.gae_coef = 0.95
        self.ppo_epoch = 10
        self.epsilon = 1e-8
        self.n_mini_batch = 4

        self.agent = PPO(self.policy, self.clip_param, self.ppo_epoch, self.n_mini_batch,
                         self.value_loss_coef, self.entropy_coef, self.learning_rate, self.epsilon,
                         self.max_grad_norm)

        self.file_metrics = './models/metrics.csv'
        self.file_winrates = './models/winrates.csv'

        if reset_files:
            if os.path.exists(self.file_metrics):
                os.remove(self.file_metrics)

            if os.path.exists(self.file_winrates):
                os.remove(self.file_winrates)

            with open(self.file_metrics, 'a') as metrics:
                metrics.write(
                    '{},{},{},{},{},{},{}\n'.format('episode', 'n_agents', 'value_loss_avg', 'value_loss_median',
                                                    'action_loss_avg', 'action_loss_median', 'reward_avg'))

            with open(self.file_winrates, 'a') as winrates:
                winrates.write(
                    '{},{},{}\n'.format('episode', 'vs_random', 'vs_fixed'))

    def run(self):
        config.verbose = {key: False for key in config.verbose}

        self.policy.base.train()

        for eps in range(self.episodes + 1):

            full_games_counter = 0
            storages1 = [Storage(20000, config.state_space, config.action_space) for _ in range(3)]
            storages2 = [Storage(20000, config.state_space, config.action_space) for _ in range(3)]

            for s in storages1:
                s.to(config.device)

            for s in storages2:
                s.to(config.device)

            game_copy = None

            for n_game in range(self.n_games):

                self.policy.update_decay()

                self.policy.base.eval()
                self.policy.use_decay = True

                print('---GAME {} / {}'.format(n_game, self.n_games))

                # n_fixed_agents = randint(1, 2)
                # n_rl_agents = randint(1, 2)
                n_fixed_agents = 1
                n_rl_agents = 1
                players = []
                rl_agents = [Player(policy=self.policy, player_id=str(idx) + '_rl', storage=storages1[idx]) for idx in range(n_rl_agents)]
                fixed_agents = [Player(policy=FixedAgent(high=randint(300, 400), low=randint(100, 200), jail=randint(50, 150)),
                                       player_id=str(idx) + '_fixed', storage=storages2[idx]) for idx in range(n_fixed_agents)]
                players.extend(rl_agents)
                players.extend(fixed_agents)
                shuffle(players)
                print('----- Players: {} fixed, {} rl'.format(n_fixed_agents, n_rl_agents))

                game = Game(players=players)
                game_copy = game

                for player in players:
                    player.set_game(game, n_game)

                for n_round in range(self.n_rounds):
                # while True:

                    # TODO: change this, don't like three completely the same conditional statements
                    if not game.is_game_active():     # stopping rounds loop
                        break

                    game.update_round()

                    for player in game.players:

                        player.reset_mortgage_buy()

                        if player.is_bankrupt:            # must change it. do it two times because some players can go bankrupt when must pay bank interest
                            game.remove_player(player)    # other player's mortgaged spaces
                            break

                        if not game.is_game_active():  # stopping players loop
                            break

                        game.pass_dice()

                        while True:
                            if not game.is_game_active():  # stopping players loop
                                break

                            player.optional_actions()

                            player.reset_mortgage_buy()

                            game.dice.roll()

                            if player.is_in_jail():
                                stay_in_jail = player.jail_strategy(dice=game.dice)
                                if stay_in_jail:
                                    player.optional_actions()
                                    break

                            if game.dice.double_counter == 3:
                                player.go_to_jail()
                                break

                            player.move(game.dice.roll_sum)

                            if player.position == 30:
                                player.go_to_jail()
                                break

                            # TODO: add card go to jail

                            space = game.board[player.position]

                            player.act(space)

                            if player.is_bankrupt:
                                game.remove_player(player)
                                break

                            if game.dice.double:
                                continue


                            # end turn
                            break

                if game.players_left == 1:
                    full_games_counter += 1

            value_losses = []
            action_losses = []
            dist_entropies = []

            self.policy.base.train()
            self.policy.use_decay = False

            for player in game_copy.players:
                if 'rl' in player.id:
                    self.update(player, value_losses, action_losses, dist_entropies)

            for player in game_copy.lost_players:
                if 'rl' in player.id:
                    self.update(player, value_losses, action_losses, dist_entropies)

            rewards = []
            for player in game_copy.players:
                if 'rl' in player.id:
                    rewards.append(player.storage.get_mean_reward())

            for player in game_copy.lost_players:
                if 'rl' in player.id:
                    rewards.append(player.storage.get_mean_reward())

            with open(self.file_metrics, 'a') as metrics:
                metrics.write(
                    '{},{},{},{},{},{},{}\n'.format(eps, n_rl_agents, np.average(value_losses), np.median(value_losses),
                                                    np.average(action_losses), np.median(action_losses), np.mean(rewards)))

            if eps % self.verbose_eval == 0:
                self.policy.base.eval()
                print('------Arena')
                arena = Arena(n_games=self.n_eval_games, n_rounds=self.n_rounds, verbose=0)  # add 3 types of logging. 0 - only show win rates.
                print('--------RL vs Random')
                winrate_random = arena.fight(agent=self.policy, opponent=RandomAgent(), opp_id='random')
                print('--------RL vs Fixed')
                winrate_fixed = arena.fight(agent=self.policy, opponent=FixedAgent(high=350, low=150, jail=100), opp_id='fixed')

                with open(self.file_winrates, 'a') as winrates:
                    winrates.write(
                        '{},{},{}\n'.format(eps, winrate_random, winrate_fixed))

            if eps % self.checkpoint_step == 0:
                torch.save(self.policy, os.path.join('models', 'model-{}.pt'.format(eps)))

            print('---Full games {} / {}'.format(full_games_counter, self.n_games))


    def update(self, player, value_losses, action_losses, dist_entropies):
        player.storage.truncate()
        player.storage.to(self.device)

        player.policy.base.eval()
        with torch.no_grad():
            next_value = player.policy.get_value(player.storage.obs[-1]).detach()

        player.storage.compute_returns(next_value, self.discount, self.gae_coef)
        value_loss, action_loss, dist_entropy = self.agent.update(player.storage)

        value_losses.append(value_loss)
        action_losses.append(action_loss)
        dist_entropies.append(dist_entropy)
