import logging
import pandas as pd
import numpy as np

from . import config
from . import bank
from . import spaces
from . import dice

logger = logging.getLogger(__name__)

class Game:

    def __init__(self, players):
        self.round = 0
        self.players = players
        self.bank = bank.Bank()
        self.board = []

        self.dice = None

        self.players_left = len(players)
        self.lost_players = []

        self.get_board(config.board_filename)
        self.update_player_indexes()

    def remove_player(self, loser):
        self.lost_players.append(self.players[loser.index])
        del self.players[loser.index]
        self.players_left = len(self.players)
        self.update_player_indexes()

        logger.info('Player {} was removed from game. Players left {}'.format(loser.id, self.players_left))

    def update_player_indexes(self):
        for i in range(self.players_left):
            self.players[i].index = i

    def pass_dice(self):
        self.dice = dice.Dice()

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

        if bid_leader:
            bid_leader.buy(space, auction_price=max_bid)


    def get_opponents(self, player):
        return [opp for opp in self.players if opp.id != player.id]

    def get_state(self, player):
        opponents = self.get_opponents(player)

        state = self.get_properties_state(player, opponents)

        position = player.position / 39

        player_properties = self.count_owned_properties(player)
        all_properties = sum([self.count_owned_properties(opponent) for opponent in opponents]) + player_properties
        properties_value = 0 if all_properties == 0 else player_properties / all_properties
        money = self.get_money(player, opponents)

        state.extend([position, properties_value, money])

        return np.array(state)

    def get_properties_state(self, player, opponents):
        properties_state = []
        for monopoly in config.monopolies:
            player_makes, player_owns = self.get_state_for_monopoly(player, monopoly)
            opponents_state = np.array([self.get_state_for_monopoly(opp, monopoly) for opp in opponents])
            opponents_make, opponents_own = opponents_state.sum(axis=0)
            properties_state.extend([player_makes, opponents_make, player_owns, opponents_own])
        return properties_state

    def get_state_for_monopoly(self, player, monopoly):
        player_makes = 0
        player_owns = 0
        if monopoly in player.properties:
            properties = player.properties[monopoly]
            if monopoly == 'Railroad':
                player_makes = properties[0].current_rent / properties[0].monopoly_max_income
                player_owns = len(properties) / 4
            elif monopoly == 'Utility':
                player_makes = properties[0].current_rent * len(properties) / properties[0].monopoly_max_income
                player_owns = len(properties) / 2
            else:
                buildings_cost = 0
                streets_cost = 0
                for property in properties:
                    streets_cost += property.price
                    buildings_cost += property.n_buildings * property.build_cost
                    player_makes += property.current_rent
                player_makes /= properties[0].monopoly_max_income
                player_owns = (streets_cost + buildings_cost) / properties[0].monopoly_max_price
        return player_makes, player_owns

    def count_owned_properties(self, player):
        return sum([len(player.properties[key]) for key in player.properties])

    def get_money(self, player, opponents):
        # print('player money', player.cash)
        # for i in range(len(opponents)):
        #     print('opp {} money {}'.format(i, opponents[i].cash))
        all_money = sum([opp.cash for opp in opponents]) + player.cash
        return player.cash / all_money

    def get_reward(self, player, state, c=0.01):
        opponents = self.get_opponents(player)

        v = self.get_make_delta(state)
        p = self.players_left
        m = self.get_money(player, opponents)
        reward = ((v / p) * c) / (1 + np.abs(v / p * c)) + m / p
        return reward

    def get_make_delta(self, player):  # return difference between what player makes from his properties
        agent_makes = 0                # and what opponents make from their properties
        opponents_make = 0
        for i in range(0, config.state_len - 3 - 1, 4):
            agent_makes += state[i]
            opponents_make += state[i + 1]
        return agent_makes - opponents_make
