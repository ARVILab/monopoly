import logging
import pandas as pd
import numpy as np
import torch

import config
from . import bank
from . import spaces
from . import dice

logger = logging.getLogger(__name__)

class Game:

    def __init__(self, players, max_rounds=300):
        self.round = 0
        self.max_rounds = max_rounds
        self.players = players
        self.bank = bank.Bank()
        self.board = []

        self.device = config.device

        self.dice = None

        self.players_left = len(players)
        self.lost_players = []

        self.get_board(config.board_filename)
        self.update_player_indexes()

        with open('rolls.npy', 'rb') as f:
            self.rolls = list(np.load(f))

    def remove_player(self, loser):
        self.lost_players.append(self.players[loser.index])
        del self.players[loser.index]
        self.players_left = len(self.players)
        self.update_player_indexes()

        # logger.info('Player {} was removed from game. Players left {}'.format(loser.id, self.players_left))

    def update_player_indexes(self):
        for i in range(self.players_left):
            self.players[i].index = i

    def pass_dice(self):
        roll = self.rolls[0]
        del self.rolls[0]
        self.dice = dice.Dice(roll)
        # self.dice = dice.Dice()

    def update_round(self):
        self.round += 1

        if config.verbose['round']:
            logger.info('\n')
            logger.info('Starting round {round}...'.format(round=self.round))

    def is_game_active(self):
        if config.verbose['is_game_active']:
            logger.info('\n')
            logger.info('{} players left, game is active {}'.format( self.players_left, self.players_left > 1))

        return self.players_left > 1

    def get_board(self, board_file):
        """
        Create board game with properties from CSV file in board_file.
        :param str board_file: Filename of CSV with board parameters
        """

        board_df = pd.read_csv(board_file)

        for _, attributes in board_df.iterrows():

            if attributes['class'] == 'Street':
                self.board.append(spaces.Street(attributes))

            if attributes['class'] == 'Railroad':
                self.board.append(spaces.Railroad(attributes))

            if attributes['class'] == 'Utility':
                self.board.append(spaces.Utility(attributes))

            if attributes['class'] == 'Tax':
                self.board.append(spaces.Tax(attributes))

            if attributes['class'] == 'Chance':
                self.board.append(spaces.Chance(attributes))

            if attributes['class'] == 'Chest':
                self.board.append(spaces.Chest(attributes))

            if attributes['class'] == 'Jail':
                self.board.append(spaces.Jail(attributes))

            if attributes['class'] == 'Idle':
                self.board.append(spaces.Idle(attributes))


    def auction(self, start_player, space):
        if config.verbose['auction']:
            logger.info('Auction begins')
        bidder_index = start_player.index
        max_bid = 0
        bid_leader = None
        org_price = space.price
        while True:
            bidder_index += 1
            if bidder_index == self.players_left:
                bidder_index -= self.players_left

            curr_bidder = self.players[bidder_index]

            if curr_bidder.id == start_player.id and not bid_leader:
                break

            if bid_leader:
                if bid_leader.id == curr_bidder.id:
                    break

            do_nothing, bid = curr_bidder.get_bid(max_bid, org_price, self.get_state(curr_bidder)) # check here whether he can afford it
            if do_nothing:
                continue

            if bid > max_bid:
                max_bid = bid
                bid_leader = curr_bidder

            if config.verbose['auction_process']:
                if bid_leader:
                    logger.info('Bid leader Player {}. Max bid {}'.format(bid_leader.id, max_bid))
                else:
                    logger.info('No bid leader')

        if config.verbose['auction']:
            logger.info('Auction finished.')
            if bid_leader:
                logger.info('Winner Player {}. Price {}'.format(bid_leader.id, max_bid))
            else:
                logger.info('No bid leader')

        if bid_leader:
            bid_leader.buy(space, auction_price=max_bid)


    def get_opponents(self, player):
        return [opp for opp in self.players if opp.id != player.id]


    def get_state(self, player):
        opponents = self.get_opponents(player)

        board_payments = self.get_board_payments(player) / 10000.
        monopolies_state = self.get_monopolies_state(player, opponents)
        player_money = player.cash / 10000.
        opponents_money = (sum([opp.cash for opp in opponents]) / len(opponents)) / 10000. if len(opponents) != 0 else 0
        player_position = np.round(player.position / 39, 2)
        opponents_position = [np.round(opp.position / 39, 2) for opp in opponents] if len(opponents) != 0 else [0]
        is_in_jail = float(player.jail_turns >= 1)
        round = np.round(self.round / self.max_rounds, 5)

        state = []
        state.extend(board_payments)
        state.extend(monopolies_state)
        state.extend([player_money, opponents_money, player_position])
        state.extend(opponents_position)
        state.extend([is_in_jail])
        state.extend([round])

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return state


    def get_board_payments(self, player):
        payments = np.zeros(40)
        for field in self.board:
            position = field.position
            type = field.type
            if type == 'Idle':
                if position == 0:
                    payments[position] = 200
                else:
                    payments[position] = 0
            elif type == 'Tax':
                if position == 4:
                    payments[position] = -200
                elif position == 38:
                    payments[position] = -100
            elif type == 'Jail' or type == 'Chest' or type == 'Chance':
                payments[position] = 0
            else:
                rent = 0
                if field.owner:
                    if field.owner.id != player.id and not field.is_mortgaged:
                        rent = -field.current_rent
                payments[position] = rent
        return payments


    def get_monopolies_state(self, player, opponents):
        properties_state = []
        for monopoly in config.monopolies:
            _, player_owns = self.get_state_for_monopoly(player, monopoly)
            if len(opponents) >= 1:
                opponents_state = np.array([self.get_state_for_monopoly(opp, monopoly) for opp in opponents])
                _, opponents_own = opponents_state.sum(axis=0)
            else:
                opponents_make, opponents_own = 0, 0
            properties_state.extend([player_owns, opponents_own])
        return properties_state


    def get_state_for_monopoly(self, player, monopoly):
        player_makes = 0
        player_owns = 0
        if monopoly in player.properties:
            properties = player.properties[monopoly]
            if monopoly == 'Railroad' or monopoly == 'Utility':
                player_makes, player_owns = self.railroads_and_utility_value(properties)
            else:
                buildings_cost = 0
                streets_cost = 0
                for property in properties:
                    if property.is_mortgaged:
                        streets_cost += property.price / 2
                    else:
                        streets_cost += property.price
                        buildings_cost += property.n_buildings * property.build_cost
                        player_makes += property.current_rent
                player_makes /= properties[0].monopoly_max_income
                player_owns = (streets_cost + buildings_cost) / properties[0].monopoly_max_price
        return np.round(player_makes, 5), np.round(player_owns, 5)

    def railroads_and_utility_value(self, properties):
        make_value = 0
        own_value = 0
        for p in properties:
            if p.is_mortgaged:
                own_value += p.price / 2
            else:
                make_value += p.current_rent
                own_value += p.price
        make_value /= p.monopoly_max_income
        own_value /= p.monopoly_max_price
        return make_value, own_value

    def count_owned_properties(self, player):
        return np.sum([len(player.properties[key]) for key in player.properties])

    def get_money(self, player, opponents):
        all_money = np.sum([opp.cash for opp in opponents]) + player.cash
        money = 0. if all_money == 0 else np.round(player.cash / all_money, 3)
        return money

    def get_income(self, player, opponents):
        player_income = player.get_income()
        all_income = np.sum([opp.get_income() for opp in opponents]) + player_income
        income = 0. if all_income == 0 else np.round(player_income / all_income, 3)
        return income

    def get_money_diff(self, player, opponents):
        player_money_norm = self.normalize(player.cash, x_min=0, x_max=2000)
        opps_money_norm = self.normalize(opponents[0].cash, x_min=0, x_max=2000) if len(opponents) == 1 else 0
        return player_money_norm - opps_money_norm

    def get_income_diff(self, player, opponents):
        player_income_norm = self.normalize(player.get_income(), x_min=0, x_max=500)
        opps_income_norm = self.normalize(opponents[0].get_income(), x_min=0, x_max=500) if len(opponents) == 1 else 0
        return player_income_norm - opps_income_norm

    def normalize(self, x, x_min, x_max, a=-1, b=1): # value, init range min, init range max, result range min, result range max
        value = (x - x_min) / (x_max - x_min) * (b - a) + a
        value = np.clip(value, a, b)
        return value

    def get_reward(self, player, state, c=1, result=0):

        # state_tmp = state.squeeze(0)
        opponents = self.get_opponents(player)


        # v = self.get_make_delta(state_tmp)
        # p = self.players_left
        # m = self.get_money(player, opponents)
        # vpc = v / p * c
        # reward = vpc / (1 + np.abs(vpc)) + m / p
        # player.compute_total_wealth()

        # money = self.get_money(player, opponents)
        # income = self.get_income(player, opponents)
        # reward = money + income

        # reward = player.reward_wealth()
        # reward = torch.from_numpy(np.array(np.round(reward, 5))).float().to(self.device).view(1, -1)

        money_diff = self.get_money_diff(player, opponents)
        income_diff = self.get_income_diff(player, opponents)

        reward = money_diff * 0.2 + income_diff * 0.8

        # reward = result
        reward = torch.FloatTensor([reward]).unsqueeze(1).to(self.device)

        return reward

    def get_make_delta(self, state):  # returns difference between what player makes from his properties
        agent_makes = 0                # and what opponents make from their properties
        opponents_make = 0
        for i in range(0, config.state_space - 3, 4):
            agent_makes += state[i]
            opponents_make += state[i + 1]
        delta = np.round(agent_makes.item() - opponents_make.item(), 5)
        return delta


    def get_leaderboard(self):
        for player in self.players:
            player.compute_total_wealth()
        leaderboard = [player for player in self.players]
        leaderboard.sort(key=lambda x: x.total_wealth, reverse=True)
        return leaderboard
