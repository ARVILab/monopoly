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

        state = self.get_properties_state(player, opponents)

        position = np.round(player.position / 39, 2)

        player_properties = self.count_owned_properties(player)
        all_properties = sum([self.count_owned_properties(opponent) for opponent in opponents]) + player_properties
        properties_value = 0 if all_properties == 0 else np.round(player_properties / all_properties, 3)
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
        return sum([len(player.properties[key]) for key in player.properties])

    def get_money(self, player, opponents):
        all_money = sum([opp.cash for opp in opponents]) + player.cash
        return np.round(player.cash / all_money, 3)

    def get_reward(self, player, state, c=0.01):
        opponents = self.get_opponents(player)

        v = self.get_make_delta(state)
        p = self.players_left
        m = self.get_money(player, opponents)
        print('v = {}, p = {}, m = {}'.format(v, p, m))
        vpc = np.abs(v / p * c)
        reward = vpc / (1 + vpc) + m / p
        return np.round(reward, 5)

    def get_make_delta(self, state):  # returns difference between what player makes from his properties
        agent_makes = 0                # and what opponents make from their properties
        opponents_make = 0
        for i in range(0, config.state_space - 3, 4):
            agent_makes += state[i]
            opponents_make += state[i + 1]
        delta = np.round(agent_makes - opponents_make, 5)
        print('Agent makes {}, opps make {}, diff {}'.format(agent_makes, opponents_make, delta))
        return delta


    def get_leaderboard(self):
        for player in self.players:
            player.compute_total_wealth()
        leaderboard = [player for player in self.players]
        leaderboard.sort(key=lambda x: x.total_wealth, reverse=True)
        return leaderboard
