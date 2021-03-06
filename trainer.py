from random import shuffle, randint
import sys
import logging
import os
import numpy as np
from tqdm import tqdm

import torch

from policies.fixed import FixedAgent
from policies.random import RandomAgent
from arena import Arena
from optimizers.ppo_optimizer import PPO
from optimizers.supervised_learning import SupervisedLearning
from optimizers.dqn_optimizer import DQNOptimizer
from nn_wrapper import NNWrapper
from monopoly.player import Player
import config
from monopoly.game import Game

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, policy, storage_class, n_episodes=100, n_games_per_eps=10, n_rounds=200, n_eval_games=50,
                 verbose_eval=50, checkpoint_step=10, reset_files=True, train_on_fixed=True, self_play=False):
        self.policy = policy
        self.n_games = n_games_per_eps
        self.n_rounds = n_rounds
        self.verbose_eval = verbose_eval
        self.n_eval_games = n_eval_games
        self.checkpoint_step = checkpoint_step
        self.device = config.device
        self.storage_class = storage_class

        self.episodes = n_episodes
        self.learning_rate = 3e-4
        self.clip_param = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.05
        self.alpha = 0.99
        self.max_grad_norm = 0.5
        self.discount = 0.99
        self.gae_coef = 0.95
        self.learning_epochs = 10
        self.epsilon = 1e-8
        self.mini_batch_size = 16384

        self.e_decay = 0.005
        self.target_update = 10


        self.train_on_fixed = train_on_fixed
        self.self_play = self_play

        if self.train_on_fixed:
            self.optimizer = SupervisedLearning(self.policy, self.mini_batch_size, self.learning_epochs,
                                                self.value_loss_coef, self.learning_rate)
        else:
            if self.policy.policy_name == 'actor_critic':
                self.optimizer = PPO(self.policy, self.clip_param, self.learning_epochs, self.mini_batch_size,
                                     self.value_loss_coef, self.entropy_coef, self.learning_rate, self.max_grad_norm)
            elif self.policy.policy_name == 'dqn':
                self.target_policy = NNWrapper('dqn', config.state_space, config.action_space, config.train_on_fixed)
                self.target_policy.to(config.device)
                self.target_policy.load_state_dict(self.policy.state_dict())
                self.optimizer = DQNOptimizer(self.policy, self.target_policy, self.mini_batch_size,
                                              self.discount, self.learning_rate, self.learning_epochs)



        self.file_metrics = './models/metrics.csv'
        self.file_winrates = './models/winrates.csv'

        if reset_files:
            if os.path.exists(self.file_metrics):
                os.remove(self.file_metrics)

            if os.path.exists(self.file_winrates):
                os.remove(self.file_winrates)

            with open(self.file_metrics, 'a') as metrics:
                metrics.write(
                    '{},{},{}\n'.format('episode', 'n_agents', 'loss_avg'))

            with open(self.file_winrates, 'a') as winrates:
                winrates.write(
                    '{},{},{}\n'.format('episode', 'vs_random', 'vs_fixed'))

    def run(self):
        config.verbose = {key: False for key in config.verbose}

        get_epsilon = lambda episode: np.exp(-episode * self.e_decay)

        for eps in range(self.episodes + 1):

            full_games_counter = 0

            game_copy = None

            storage1 = self.storage_class()
            storage2 = self.storage_class()

            if self.train_on_fixed:
                self.policy.train_on_fixed = True

            print('---STARTING SIMULATIONS')
            for n_game in tqdm(range(self.n_games)):

                n_opps_agents = 1
                n_rl_agents = 1
                players = []

                rl_agents = [
                    Player(policy=self.policy, player_id=str(idx) + '_rl', storage=storage1) for idx in range(n_rl_agents)]

                if self.policy.policy_name == 'dqn':
                    self.policy.policy.update_epsilon(get_epsilon(eps))

                if not self.self_play:
                    opp_agents = [
                        Player(policy=FixedAgent(high=350, low=150, jail=100),
                               player_id=str(idx) + '_fixed', storage=self.storage_class()) for idx in
                        range(n_opps_agents)]
                else:
                    opp_agents = [
                        Player(policy=self.policy, player_id=str(idx + 1) + '_rl', storage=storage2) for idx in range(n_rl_agents)]

                players.extend(rl_agents)
                players.extend(opp_agents)
                # shuffle(players)
                # print('----- Players: {} fixed, {} rl'.format(n_fixed_agents, n_rl_agents))

                game = Game(players=players, max_rounds=self.n_rounds)
                game_copy = game

                for player in players:
                    player.set_game(game, n_game)

                game_finished = False

                for n_round in range(self.n_rounds):
                    if game_finished:
                        break

                    game.update_round()

                    for player in game.players:
                        if not game.is_game_active():  # stopping rounds loop
                            player.won()
                            game_finished = True
                            break

                        # player.reset_mortgage_buy()

                        if player.is_bankrupt:            # must change it. do it two times because some players can go bankrupt when must pay bank interest
                            game.remove_player(player)    # other player's mortgaged spaces
                            break

                        game.pass_dice()

                        while True:
                            if not game.is_game_active():  # stopping players loop
                                break

                            player.optional_actions()

                            # player.reset_mortgage_buy()

                            game.dice.roll()

                            if player.is_in_jail():
                                stay_in_jail = player.jail_strategy(dice=game.dice)
                                if stay_in_jail:
                                    player.optional_actions()
                                    break

                            if player.is_bankrupt:
                                game.remove_player(player)
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
                else:
                    for player in game.players:
                       player.draw()

            losses = []

            for player in game_copy.players:
                if 'rl' in player.id:
                    self.update(player, losses)

            for player in game_copy.lost_players:
                if 'rl' in player.id:
                    self.update(player, losses)


            if eps % self.target_update == 0:
                self.target_policy.load_state_dict(self.policy.state_dict())

            rewards = []
            for player in game_copy.players:
                if 'rl' in player.id:
                    rewards.append(player.storage.get_mean_reward())

            for player in game_copy.lost_players:
                if 'rl' in player.id:
                    rewards.append(player.storage.get_mean_reward())

            with open(self.file_metrics, 'a') as metrics:
                metrics.write('{},{},{}\n'.format(eps, n_rl_agents, np.average(losses)))



            if eps % self.verbose_eval == 0:
                if self.train_on_fixed:
                    self.policy.train_on_fixed = False
                print('------Arena')
                arena = Arena(n_games=self.n_eval_games, n_rounds=self.n_rounds, verbose=0)  # add 3 types of logging. 0 - only show win rates.
                print('--------RL vs Random')
                winrate_random = arena.fight(agent=self.target_policy, opponent=RandomAgent(), opp_id='random')
                print('--------RL vs Fixed')
                winrate_fixed = arena.fight(agent=self.target_policy, opponent=FixedAgent(high=350, low=150, jail=100), opp_id='fixed')

                with open(self.file_winrates, 'a') as winrates:
                    winrates.write(
                        '{},{},{}\n'.format(eps, winrate_random, winrate_fixed))

            if eps % self.checkpoint_step == 0:
                torch.save(self.target_policy, os.path.join('models', 'model-{}.pt'.format(eps)))

            print('---Full games {} / {}'.format(full_games_counter, self.n_games))

    def update(self, player, losses):
        # with torch.no_grad():
        #     next_value = player.policy.get_value(player.storage.states[-1])

        player.storage.compute()

        # player.storage.show()

        loss = self.optimizer.update(player.storage)
        losses.append(loss)

