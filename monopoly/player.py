from . import config
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Player:

    def __init__(self, policy=None, jail_policy=None, player_id=None):

        self.id = player_id             # Identification number
        self.index = 0
        self.cash = 1500                # Cash on hand
        self.properties = {}            # Dict of monopolies
        self.position = 0               # Board position
        self.jail_cards = 0             # Number of "Get Out Of Jail Free" cards
        self.jail_turns = 0             # Number of remaining turns in jail
        self.is_bankrupt = False        # Bankrupt status

        self.policy = policy
        self.storage = None

        self.obligatory_acts = {
            'Idle': None,
            'Street': self.act_on_property,
            'Railroad': self.act_on_property,
            'Utility': self.act_on_property,
            'Chance': self.act_on_chance,
            'Chest': self.act_on_chest,
            'Tax': self.act_on_tax
        }

    def set_game(self, game):
        self.game = game

    def act(self, space):
        space_type = space.get_type()
        obligation = self.obligatory_acts[space_type]

        if obligation:
            obligation(space, self.game.dice)

        if not self.is_bankrupt:
            self.optional_actions()


    def move(self, roll):
        old_position = self.position
        self.position += roll

        if self.position >= 40:
            self.position -= 40
            self.cash += 200

        if config.verbose['move']:
            logger.info('Player {id} moved on board from position {old_position} to {new_position}. Now on {space}'.format(
                id=self.id, old_position=old_position, new_position=self.position, space=self.game.board[self.position].name))

    def pay(self, money, recepient=None):
        self.cash -= money
        if recepient:
            recepient.cash += money

    def have_enough_money(self, money):
        return self.cash >= money

    def buy(self, space, auction_price=None):
        price = auction_price if auction_price else space.price
        monopoly = space.monopoly
        self.cash -= price
        if monopoly not in self.properties:
            self.properties[monopoly] = []
        self.properties[monopoly].append(space)
        space.onwer = self
        self.update_rent(space)

    def sell_all_monopoly_building(self, monopoly):
        for space in monopoly:
            self.cash += space.n_buildings * space.build_cost / 2

    def is_in_jail(self):
        if self.jail_turns == 0:
            self.pay_to_leave_jail()
            return False
        elif self.jail_turns > 0:
            self.jail_turns -= 1
            return True

    def go_to_jail(self):
        self.position = 10
        self.jail_turns = 2 # because he leaves jail on the third move

        logger.info('Player {id} went to jail'.format(id=self.id))

        self.jail_strategy()

    def unowned_street(self, space):
        mask_params = [False, False, False, False]
        action_mask = self.get_action_mask(mask_params)
        if self.have_enough_money(space.price):
            action_mask[1 + space.position_in_action] = 1.
        state = self.game.get_state(self)
        action = self.policy.act(state) # don't forget to apply mask
        action = self.apply_mask(action, action_mask)
        do_nothing = self.apply_action(action)
        if do_nothing:
            self.game.auction(self, space)

    def get_bid(self, max_bid, org_price, state):
        do_nothing, bid = self.policy.auction_policy(max_bid, org_price, state)
        return do_nothing, bid

    def jail_strategy(self, dice=None):
        mask_params = [False, False, True, False]
        action_mask = self.get_action_mask(mask_params)
        state = self.game.get_state(self)
        action = self.policy.jail_policy(state)
        action_mask[-3] = action[0]
        action_mask[-2] = action[1]
        do_nothing = self.apply_action(action) # here if do_nothing is False it means that he neither paid nor used the card
        if do_nothing:
            if dice:
                if dice.double:
                    return False # means not staying in jail
            return True # staying in jail
        return False # means he paid or used card


    def optional_actions(self):
        if config.verbose['optional_actions']:
            logger.info('\n')
            logger.info('Player {id} does optional actions'.format(id=self.id))

        while True:
            mask_params = [True, True, False, False]
            action_mask = self.get_action_mask(mask_params)
            state = self.game.get_state(self)
            action = self.policy.act(state)
            action = self.apply_mask(action, action_mask)
            do_nothing = self.apply_action(action)
            if do_nothing:
                break

    def apply_action(self, action):
        action_index = action.argmax()

        # don't like it. must change it
        if action_index == 0:
            return True # do nothing
        elif action_index > 0 and action_index < 29:
            target = action_index - 1
            self.spend_money(target)
        elif action_index > 28 and action_index < 57:
            target = action_index - 29
            self.make_money(target)
        elif action_index == 57:
            self.pay_to_leave_jail()
        elif action_index == 58:      # using the card
            pass
        elif action_index == 59:
            self.trade()
        return False


    def spend_money(self, target): # target on which player will spend money
        space = self.get_space_by_action_index(target)
        if not space:
            raise ValueError('In spend_money space is somehow none!!!')
        monopoly = []
        if space.monopoly in self.properties:
            monopoly = self.properties[space.monopoly]
        if self.can_buy_space(space):
            self.buy(space)
        elif self.can_unmortgage(space):
            self.unmortgage(space)
        elif self.can_build_on_monopoly(monopoly):
            if self.can_build_on_space(space, monopoly):
                self.buy_building(space)

    def make_money(self, target):  # target from which player will make money
        space = self.get_space_by_action_index(target)
        if not space:
            raise ValueError('In make_money space is somehow none!!!')
        monopoly = []
        if space.monopoly in self.properties:
            monopoly = self.properties[space.monopoly]
        if self.can_sell_building(space, monopoly):
            self.sell_building(space)
        elif self.can_mortgage(space, monopoly):
            self.mortgage(space)

    def trade(self):
        pass

    def pay_to_leave_jail(self):  # handle bankruptcy here
        self.jail_turns = 0
        money_owned = 50
        if self.have_enough_money(money_owned):
            self.pay(money_owned)
        else:
            self.try_to_survive()
            if self.have_enough_money(money_owned):
                self.pay(money_owned)
            else:
                self.go_bankrupt()


    def unmortgage(self, space):
        self.cash -= space.price_mortgage + space.price * 0.1
        space.is_mortgaged = False
        self.update_rent(space)

    def buy_building(self, space):
        self.cash -= space.build_cost
        space.n_buildings += 1
        self.update_building_rent(space)

    def sell_building(self, space):
        self.cash += space.build_cost
        space.n_buildings -= 1
        self.update_building_rent(space)

    def mortgage(self, space):
        self.cash += space.price_mortgage
        space.is_mortgaged = True
        self.update_rent(space)

    def get_space_by_action_index(self, action_index):
        for space in self.game.board:
            if action_index == space.position_in_action:
                return space
        return None

    def apply_mask(self, action, mask):
        return np.array(action) * np.array(mask)

    def can_build_on_monopoly(self, monopoly):
        if len(monopoly) != 0:
            if monopoly[0].get_type() == 'Street':
                if len(monopoly) == monopoly[0].monopoly_size:
                    return True
        return False

    def can_build_on_space(self, space, monopoly):
        if len(monopoly) != 0:
            if monopoly[0].get_type() == 'Street':
                if space.n_buildings < 5:
                    buildings_list = [s.n_buildings for s in monopoly]
                    max_buildings = np.max(buildings_list)
                    all_equal = buildings_list[1:] == buildings_list[:-1]
                    if space.n_buildings != max_buildings or space.n_buildings == 0 or all_equal:
                        if self.have_enough_money(space.build_cost):
                            return True
        return False

    def can_sell_building(self, space, monopoly):
        if len(monopoly) != 0:
            if monopoly[0].get_type() == 'Street':
                if space.n_buildings > 0:
                    buildings_list = [s.n_buildings for s in monopoly]
                    max_buildings = np.max(buildings_list)
                    if space.n_buildings == max_buildings:
                        return True
        return False

    def can_mortgage(self, space, monopoly):
        if not space.is_mortgaged:
            if monopoly[0].get_type() == 'Street':
                buildings_list = [s.n_buildings for s in monopoly]
                max_buildings = np.max(buildings_list)
                if max_buildings == 0 and not space.is_mortgaged:
                    return True
                return False
            return True
        return False

    def can_unmortgage(self, space):
        if space.is_mortgaged:
            if self.have_enough_money(space.price_mortgage + space.price_mortgage * 0.1):
                return True
        return False

    def can_buy_space(self, space):
        if not space.owner and self.position == space.position:
            if self.have_enough_money(space.price):
                return True
        return False


    def get_action_mask(self, params):
        mask = [1.]                                             # do nothing is always available
        spend_money_mask = self.get_spend_money_mask(params[0]) # mask for buying streets and building on those streets, and unmortgaging
        make_money_mask = self.get_make_money_mask(params[1])   # mask for mortgaging streets and selling building from those streets
        jail_mask = self.get_jail_mask(params[2])               # action to take when in jail
        trading_mask = self.get_trading_mask(params[3])         # trading mask. zero for now
        mask.extend(spend_money_mask)
        mask.extend(make_money_mask)
        mask.extend(jail_mask)
        mask.extend(trading_mask)
        return np.array(mask)

    def get_spend_money_mask(self, is_avaliable):
        mask = np.zeros(28)
        if is_avaliable:
            for key in self.properties:
                monopoly = self.properties[key]
                build_on_monopoly = self.can_build_on_monopoly(monopoly)
                for space in monopoly:
                    if self.can_unmortgage(space):
                        mask[space.position_in_action] = 1.
                    elif build_on_monopoly:
                        if self.can_build_on_space(space, monopoly):
                            mask[space.position_in_action] = 1.
        return mask

    def get_make_money_mask(self, is_avaliable):
        mask = np.zeros(28)
        if is_avaliable:
            for key in self.properties:
                monopoly = self.properties[key]
                for space in monopoly:
                    if self.can_sell_building(space, monopoly):
                        mask[space.position_in_action] = 1.
                    elif self.can_mortgage(space, monopoly):
                        mask[space.position_in_action] = 1.
        return mask

    def get_jail_mask(self, is_avaliable):
        mask = np.zeros(2)
        if is_avaliable:
            if self.have_enough_money(50):
                mask[0] = 1.
            if self.jail_cards > 0:
                mask[1] = 1.
        return mask

    def get_trading_mask(self, is_avaliable):
        mask = np.zeros(1)
        if is_avaliable:
            mask[0] = 1.
        return mask

    def act_on_property(self, space, dice):
        if space.owner:
            money_owned = space.get_rent(dice.roll_sum)
            if self.have_enough_money(money_owned):
                self.pay(money_owned, space.owner)
            else:
                self.try_to_survive()
                if self.have_enough_money(money_owned):
                    self.pay(money_owned, space.owner)
                else:
                    self.go_bankrupt(space.owner)
        else:
            self.unowned_street(space)


    def act_on_chance(self, space, dice):
        pass

    def act_on_chest(self, space, dice):
        pass

    def act_on_tax(self, space, dice):
        money_owned = space.tax
        if self.have_enough_money(money_owned):
            self.pay(money_owned)
        else:
            self.try_to_survive()
            if self.have_enough_money(money_owned):
                self.pay(money_owned)
            else:
                self.go_bankrupt()

    def count_mortgaged(self, monopoly):
        return len([space for space in monopoly if space.is_mortgaged])

    def update_rent(self, space):
        monopoly = self.properties[space.monopoly]
        n_owned = len(monopoly)
        n_mortgaged = self.count_mortgaged(monopoly)

        diff = n_owned - n_mortgaged
        space_type = space.get_type()

        if space_type == 'Street':
            if space.monopoly_size == diff:
                for s in monopoly:
                    s.current_rent = space.monopoly_rent
        if space_type == 'Railroad':
            if diff == 1:
                space.current_rent = space.init_rent
            if diff == 2:
                for s in monopoly:
                    s.current_rent = space.rent_railroad_2
            if diff == 3:
                for s in monopoly:
                    s.current_rent = space.rent_railroad_3
            if diff == 4:
                for s in monopoly:
                    s.current_rent = space.monopoly_rent
        if space_type == 'Utility':
            if diff == 1:
                space.current_rent = space.init_rent
            if diff == 2:
                for s in monopoly:
                    s.current_rent = space.monopoly_rent

    def update_building_rent(self, space):
        n_buildings = space.n_buildings
        if n_buildings == 0:
            space.rent_now = space.init_rent
        if n_buildings == 1:
            space.rent_now = space.rent_house_1
        if n_buildings == 2:
            space.rent_now = space.rent_house_2
        if n_buildings == 3:
            space.rent_now = space.rent_house_3
        if n_buildings == 4:
            space.rent_now = space.rent_house_4
        if n_buildings == 5:
            space.rent_now = space.rent_hotel

    def sell_all_buildings(self):
        for monopoly in self.properties:
            if monopoly[0].get_type() == 'Street':
                self.sell_all_monopoly_building(monopoly)

    def go_bankrupt(self, creditor=None):
        if creditor:
            creditor.jail_cards += self.jail_cards
            self.sell_all_buildings()
            creditor.cash += self.cash
            went_bankrupt = False
            for monopoly in self.properties:
                for space in monopoly:
                    if space.monopoly not in creditor.properties:
                        creditor.properties[space.monopoly] = []
                    creditor.properties[space.monopoly].append(space)
                    space.owner = creditor
                    if space.is_mortgaged:
                        creditor.pay_bank_interest(space)
                        if not creditor.is_bankrupt:
                            went_bankrupt = True
                            break
                        mask_params = [True, False, False, False]
                        action_mask = creditor.get_action_mask(mask_params)
                        state = self.game.get_state(creditor)
                        action = creditor.policy.act(state) # don't forget to apply mask
                        action = creditor.apply_mask(action, action_mask)
                        do_nothing = creditor.apply_action(action)
                    creditor.update_rent(space)
                if went_bankrupt:
                    break
        else:
            for key in self.properties:
                for space in self.properties[key]:
                    space.nullify()
                    self.game.auction(self, space)
        self.cash = 0
        self.jail_cards = 0
        self.properties = {}
        self.bankrupt = True

    def pay_bank_interest(self, space, creditor):
        bank_interest = space.price * 0.1
        if creditor.have_enough_money(bank_interest):
            creditor.pay(bank_interest)
        else:
            creditor.try_to_survive()
            if creditor.have_enough_money(bank_interest):
                creditor.pay(bank_interest)
            else:
                creditor.go_bankrupt()

    def try_to_survive(self,):
        while True:
            mask_params = [False, True, False, False]
            action_mask = self.get_action_mask(mask_params)
            state = self.game.get_state(self)
            action = self.policy.act(state)
            action = self.apply_mask(action, action_mask)
            do_nothing = self.apply_action(action)
            if do_nothing:
                break
