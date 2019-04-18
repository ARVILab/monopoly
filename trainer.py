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

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, policy, n_games=500, n_rounds=200, n_eval_games=50, verbose_eval=50, checkpoint_step=10, reset_files=True):
        self.policy = policy
        self.n_games = n_games
        self.n_rounds = n_rounds
        self.verbose_eval = verbose_eval
        self.n_eval_games = n_eval_games
        self.checkpoint_step = checkpoint_step
        self.device = config.device

        self.learning_rate = 1e-4
        self.clip_param = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.1
        self.alpha = 0.99
        self.max_grad_norm = 0.5
        self.discount = 0.99
        self.gae_coef = 0.95
        self.ppo_epoch = 10
        self.epsilon = 1e-8
        self.n_mini_batch = 4

        self.agent = PPO(self.policy, self.clip_param, self.ppo_epoch,
                         self.value_loss_coef, self.entropy_coef, self.learning_rate, self.epsilon, self.max_grad_norm,
                         self.n_mini_batch)

        self.file_metrics = './models/metrics.csv'
        self.file_winrates = './models/winrates.csv'

        if reset_files:
            if os.path.exists(self.file_metrics):
                os.remove(self.file_metrics)

            if os.path.exists(self.file_winrates):
                os.remove(self.file_winrates)

            with open(self.file_metrics, 'a') as metrics:
                metrics.write(
                    '{},{},{},{},{},{}\n'.format('episode', 'n_agents', 'value_loss_avg', 'value_loss_median',
                                                 'action_loss_avg', 'action_loss_median'))

            with open(self.file_winrates, 'a') as winrates:
                winrates.write(
                    '{},{},{}\n'.format('episode', 'vs_random', 'vs_fixed'))


    def run(self):
        config.verbose = {key: False for key in config.verbose}

        full_games_counter = 0

        for n_game in range(self.n_games):

            self.policy.base.eval()

            print('---GAME {} / {}'.format(n_game, self.n_games))

            # n_players = randint(2, 5)
            n_players = 2
            players = [Player(policy=self.policy, player_id=str(idx) + 'rl') for idx in range(n_players)]
            shuffle(players)

            game = Game(players=players)

            for player in players:
                player.set_game(game)

            for n_round in range(self.n_rounds):

                # TODO: change this, don't like three completely the same conditional statements
                if not game.is_game_active():     # stopping rounds loop
                    break

                game.update_round()

                for player in game.players:

                    if player.is_bankrupt:            # must change it. do it two times because some players can go bankrupt when must pay bank interest
                        game.remove_player(player)    # other player's mortgaged spaces
                        break

                    if not game.is_game_active():  # stopping players loop
                        break

                    game.pass_dice()

                    while True:
                        if not game.is_game_active():  # stopping players loop
                            break

                        if n_round != 0:
                            player.optional_actions()

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

            value_losses = []
            action_losses = []
            dist_entropies = []

            self.policy.base.train()

            for player in game.players:
                self.update(player, value_losses, action_losses, dist_entropies)

            for player in game.lost_players:
                self.update(player, value_losses, action_losses, dist_entropies)

            with open(self.file_metrics, 'a') as metrics:
                metrics.write(
                    '{},{},{},{},{},{}\n'.format(n_game, n_players, np.average(value_losses), np.median(value_losses),
                                                 np.average(action_losses), np.median(action_losses)))

            if game.players_left == 1:
                full_games_counter += 1

            if n_game % self.verbose_eval == 0:
                self.policy.base.eval()
                print('------Arena')
                arena = Arena(n_games=self.n_eval_games, n_rounds=self.n_rounds, verbose=0)  # add 3 types of logging. 0 - only show win rates.
                print('--------RL vs Random')
                winrate_random = arena.fight(agent=self.policy, opponent=RandomAgent(), opp_id='random')
                print('--------RL vs Fixed')
                winrate_fixed = arena.fight(agent=self.policy, opponent=FixedAgent(high=350, low=150, jail=100), opp_id='fixed')

                with open(self.file_winrates, 'a') as winrates:
                    winrates.write(
                        '{},{},{}\n'.format(n_game, winrate_random, winrate_fixed))

            if n_game % self.checkpoint_step == 0:
                torch.save(self.policy, os.path.join('models', 'model-{}.pt'.format(n_game)))

        print('---Full games {} / {}'.format(full_games_counter, self.n_games))


    def update(self, player, value_losses, action_losses, dist_entropies):
        player.storage.truncate()
        player.storage.to(self.device)

        player.policy.base.eval()
        with torch.no_grad():
            next_value = player.policy.get_value(player.storage.obs[-2]).detach()

        player.storage.compute_returns(next_value, self.discount, self.gae_coef)
        value_loss, action_loss, dist_entropy = self.agent.update(player.storage)

        value_losses.append(value_loss)
        action_losses.append(action_loss)
        dist_entropies.append(dist_entropy)
