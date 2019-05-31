import config

import numpy as np
import logging

import torch

logger = logging.getLogger(__name__)

class Player:
    def __init__(self, policy=None, player_id=None, storage=None):

        self.id = player_id             # Identification number
        self.index = 0
        self.cash = 1500                # Cash on hand
        self.properties = {}            # Dict of monopolies
        self.position = 0               # Board position
        self.jail_cards = 0             # Number of "Get Out Of Jail Free" cards
        self.jail_turns = 0             # Number of remaining turns in jail
        self.is_bankrupt = False        # Bankrupt status

        self.policy = policy
        self.storage = storage
        self.device = config.device

        self.mortgages = []
        self.buyings = []
        self.last_state = None
        self.last_reward = None
        self.game = None

        self.obligatory_acts = {
            'Idle': None,
            'Street': self.act_on_property,
            'Railroad': self.act_on_property,
            'Utility': self.act_on_property,
            'Chance': self.act_on_chance,
            'Chest': self.act_on_chest,
            'Tax': self.act_on_tax
        }

    def show(self):
        print('Player:', self.id)
        print('Position:', self.position)
        print('Money:', self.cash)
        print('Owns:')
        for monopoly in self.properties:
            print('---{}:'.format(monopoly))
            for space in self.properties[monopoly]:
                print('-----{}, rent {}'.format(space.name, space.current_rent))
                if space.get_type() == 'Street':
                    if space.n_buildings != 0:
                        print('----------{}'.format(space.n_buildings))
                if space.is_mortgaged:
                    print('---------------Mortgaged')

    def set_game(self, game, n_game):
        self.game = game
        self.last_state = self.game.get_state(self)
        self.last_reward = game.get_reward(self, self.last_state)

    def reset_mortgage_buy(self):
        self.mortgages = []
        self.buyings = []

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
            if config.verbose['go']:
                logger.info('Player {id} passed through Go. Got 200 bucks. Now have {cash}'.format(id=self.id, cash=self.cash))

        if config.verbose['move']:
            logger.info('Player {id} moved on board from position {old_position} to {new_position}. Now on {space}'.format(
                id=self.id, old_position=old_position, new_position=self.position, space=self.game.board[self.position].name))

    def pay(self, money, recipient=None):
        self.cash -= money
        if recipient:
            recipient.cash += money

        if config.verbose['pay']:
            recep_id = None
            if recipient:
                recep_id = recipient.id
            else:
                recep_id = 'bank'
            logger.info('Player {id1} payed Player {id2} {money}'.format(id1=self.id, id2=recep_id, money=money))

    def have_enough_money(self, money):
        return self.cash >= money

    def buy(self, space, auction_price=None):
        price = auction_price if auction_price else space.price
        monopoly = space.monopoly
        self.cash -= price
        if monopoly not in self.properties:
            self.properties[monopoly] = []
        self.properties[monopoly].append(space)
        space.owner = self
        self.update_rent(space)

        if config.verbose['buy']:
            logger.info('Player {id} bought {space_name} for {price}. Player {id} has {money} now.'.format(id=self.id, space_name=space.name,
                                                                                        price=price, money=self.cash))

    def sell_all_monopoly_building(self, monopoly):
        for space in self.properties[monopoly]:
            if space.n_buildings != 0:
                self.cash += space.n_buildings * space.build_cost / 2
                space.n_buildings = 0
                self.update_building_rent(space)

    def is_in_jail(self):
        if self.jail_turns == 1:
            if config.verbose['3_moves_in_jail']:
                print('Player {} was in jail for 3 moves.'.format(self.id))
            self.pay_to_leave_jail()
            return False
        elif self.jail_turns > 1:
            self.jail_turns -= 1
            return True

    def go_to_jail(self):
        self.position = 10
        self.jail_turns = 3 # because he leaves jail on the third move

        # logger.info('Player {id} went to jail'.format(id=self.id))

        self.jail_strategy()

    def unowned_street(self, space):
        mask_params = [False, False, False, False]
        action_mask = self.get_action_mask(mask_params)
        if self.have_enough_money(space.price):
            action_mask[space.position_in_action + 1] = 1.
        action_mask_gpu = torch.FloatTensor(action_mask).to(self.device)

        state = self.game.get_state(self)
        with torch.no_grad():
            value, action, action_log_prob = self.policy.act(state, self.cash, action_mask_gpu)

        # action_item = action.item()
        # if action_item >= 29 and action_item <= 56:
        #     self.mortgages.append(action_item)
        # if action_item >= 1 and action_item <= 28:
        #     self.buyings.append(action_item)

        do_nothing = self.apply_action(action)

        if do_nothing:
            # self.game.auction(self, space)
            pass


        next_state = self.game.get_state(self)
        reward = self.game.get_reward(self, next_state)

        last_available_action = action_mask_gpu.cpu().argmax().item()

        if last_available_action != 0:
            self.storage.push(state, action, action_log_prob, value, reward, [1.0])

    def get_bid(self, max_bid, org_price, state):
        do_nothing, bid = self.policy.auction_policy(max_bid, org_price, state, self.cash)
        if bid > self.cash:
            return False, 0
        return do_nothing, bid

    def jail_strategy(self, dice=None):
        mask_params = [False, False, True, False]
        action_mask = self.get_action_mask(mask_params)
        action_mask_gpu = torch.FloatTensor(action_mask).to(self.device)

        state = self.game.get_state(self)
        with torch.no_grad():
            value, action, action_log_prob = self.policy.jail_policy(state, self.cash, action_mask_gpu)

        do_nothing = self.apply_action(action)
        if do_nothing:
            if dice:
                if dice.double:

                    if config.verbose['double_leave_jail']:
                        logger.info('Player {} got double. Leave jail.'.format(self.id))

                    next_state = self.game.get_state(self)
                    reward = self.game.get_reward(self, next_state)

                    last_available_action = action_mask_gpu.cpu().argmax().item()
                    if last_available_action != 0:
                        self.storage.push(state, action, action_log_prob, value, reward, [1.0])

                    return False # means not staying in jail

            if config.verbose['stay_in_jail']:
                logger.info('Player {} stays in jail.'.format(self.id))

            next_state = self.game.get_state(self)
            reward = self.game.get_reward(self, next_state)

            last_available_action = action_mask_gpu.cpu().argmax().item()
            if last_available_action != 0:
                self.storage.push(state, action, action_log_prob, value, reward, [1.0])

            return True # staying in jail

        next_state = self.game.get_state(self)
        reward = self.game.get_reward(self, next_state)

        last_available_action = action_mask_gpu.cpu().argmax().item()
        if last_available_action != 0:
            self.storage.push(state, action, action_log_prob, value, reward, [1.0])

        return False # means he paid or used card


    def optional_actions(self):
        if config.verbose['optional_actions']:
            logger.info('\n')
            logger.info('Player {id} does optional actions'.format(id=self.id))

        while True:
            mask_params = [True, True, False, False]
            action_mask = self.get_action_mask(mask_params)
            action_mask_gpu = torch.FloatTensor(action_mask).to(self.device)

            state = self.game.get_state(self)
            with torch.no_grad():
                value, action, action_log_prob = self.policy.act(state, self.cash, action_mask_gpu, self.mortgages, self.buyings)

            # action_item = action.item()
            # if action_item >= 29 and action_item <= 56:
            #     self.mortgages.append(action_item)
            # if action_item >= 1 and action_item <= 28:
            #     self.buyings.append(action_item)

            do_nothing = self.apply_action(action)

            next_state = self.game.get_state(self)
            reward = self.game.get_reward(self, next_state)

            last_available_action = action_mask_gpu.cpu().argmax().item()
            if last_available_action != 0:
                self.storage.push(state, action, action_log_prob, value, reward, [1.0])

            if do_nothing:
                break

    def apply_action(self, action):
        try:
            action_item = action.item()
        except:
            print(action)
            raise Exception('WTF')

        if action_item == 0:
            return True # do nothing
        elif action_item > 0 and action_item < 29:
            target = action_item - 1
            self.spend_money(target)
        elif action_item > 28 and action_item < 57:
            target = action_item - 29
            self.make_money(target)
        elif action_item == 57:
            self.pay_to_leave_jail()
        elif action_item == 58:      # using the card
            pass
        elif action_item == 59:
            # self.trade()
            pass
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
            if config.verbose['unmortgage_ability']:
                price = space.price_mortgage + space.price * 0.1
                logger.info('Can unmortgage. Player {id} have money {cash}. Need money {price}'.format(
                    id=self.id,
                    cash=self.cash,
                    price=price
                ))
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
        if config.verbose['pay_to_leave_jail']:
            logger.info('Player {} pays to leave jail. Have money {}'.format(self.id, self.cash))
        if self.have_enough_money(money_owned):

            self.pay(money_owned)
        else:
            self.try_to_survive()
            if self.have_enough_money(money_owned):
                self.pay(money_owned)
            else:
                # logger.info('Could not pay for leaving jail after 3 move. Went bankrupt.')
                self.go_bankrupt()


    def unmortgage(self, space):
        price = space.price_mortgage + space.price * 0.1
        self.cash -= price
        space.is_mortgaged = False
        self.update_rent(space)
        if config.verbose['unmortgage']:
            logger.info('{space_name} is unmortgaged now. Player {id} pays {mortg_money} money. Player {id} has {money}'.format(space_name=space.name, id=self.id,
                                                                                                                    mortg_money=price, money=self.cash))

    def buy_building(self, space):
        self.cash -= space.build_cost
        space.n_buildings += 1
        if config.verbose['buy_building']:
            logger.info('Player {id} buys building on space {space_name}. Spent money {build_cost}. Now have money {cash}. Building on this space {n_buildings}'.format(
                id=self.id,
                space_name=space.name,
                build_cost=space.build_cost,
                cash=self.cash,
                n_buildings=space.n_buildings
            ))
        self.update_building_rent(space)

    def sell_building(self, space):
        self.cash += space.build_cost / 2
        space.n_buildings -= 1
        if config.verbose['sell_building']:
            logger.info('Player {id} sell building on space {space_name}. Got money {build_cost}. Now have money {cash}. Building on this space {n_buildings}'.format(
                id=self.id,
                space_name=space.name,
                build_cost=(space.build_cost / 2),
                cash=self.cash,
                n_buildings=space.n_buildings
            ))
        self.update_building_rent(space)

    def mortgage(self, space):
        self.cash += space.price_mortgage
        space.is_mortgaged = True
        self.update_rent(space)
        if config.verbose['mortgage']:
            logger.info('{space_name} is mortgaged now. Player {id} gets {mortg_money} money. Player {id} has {money}'.format(space_name=space.name, id=self.id,
                                                                                                                    mortg_money=space.price_mortgage, money=self.cash))

    def get_space_by_action_index(self, action_index):
        for space in self.game.board:
            if action_index == space.position_in_action:
                return space
        return None


    def can_build_on_monopoly(self, monopoly):
        for space in monopoly:
            if space.is_mortgaged:
                return False
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
            if self.have_enough_money(space.price_mortgage + space.price * 0.1):
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

    def get_make_money_mask(self, is_available):
        mask = np.zeros(28)
        if is_available:
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
            if space.owner.id == self.id:
                return
            if not space.is_mortgaged:
                money_owned = space.get_rent(dice.roll_sum)
                if self.have_enough_money(money_owned):
                    self.pay(money_owned, space.owner)
                else:
                    self.try_to_survive()
                    if self.have_enough_money(money_owned):
                        self.pay(money_owned, space.owner)
                    else:
                        if config.verbose['before_bankrupt']:
                            logger.info(
                                'Player {id2} could not pay Player {id1} a rent on space {name}. Player {id2} has {cash} but need {rent}'.format(
                                    id1=space.owner.id,
                                    name=space.name,
                                    id2=self.id,
                                    cash=self.cash,
                                    rent=space.current_rent))
                        self.go_bankrupt(space.owner)
            else:
                if config.verbose['on_mortgaged_property']:
                    logger.info('{} is mortgaged.'.format(space.name))
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
                if config.verbose['cant_pay_tax']:
                    logger.info(
                        'Player {id} could not pay tax. Player {id} has {cash} but need {tax}'.format(
                            id=self.id,
                            cash=self.cash,
                            tax=money_owned))
                self.go_bankrupt()

    def count_mortgaged(self, monopoly):
        return len([space for space in monopoly if space.is_mortgaged])

    def update_rent(self, space):
        monopoly = self.properties[space.monopoly]
        n_owned = len(monopoly)

        space_type = space.get_type()

        if space_type == 'Street':
            if space.monopoly_size == n_owned:
                for s in monopoly:
                    s.current_rent = s.monopoly_rent
        if space_type == 'Railroad':
            if n_owned == 1:
                space.current_rent = space.init_rent
            if n_owned == 2:
                for s in monopoly:
                    s.current_rent = s.rent_railroad_2
            if n_owned == 3:
                for s in monopoly:
                    s.current_rent = s.rent_railroad_3
            if n_owned == 4:
                for s in monopoly:
                    s.current_rent = s.monopoly_rent
        if space_type == 'Utility':
            if n_owned == 1:
                space.current_rent = space.init_rent
            if n_owned == 2:
                for s in monopoly:
                    s.current_rent = s.monopoly_rent


    def update_building_rent(self, space):
        n_buildings = space.n_buildings
        if n_buildings == 0:
            space.current_rent = space.monopoly_rent
        if n_buildings == 1:
            space.current_rent = space.rent_house_1
        if n_buildings == 2:
            space.current_rent = space.rent_house_2
        if n_buildings == 3:
            space.current_rent = space.rent_house_3
        if n_buildings == 4:
            space.current_rent = space.rent_house_4
        if n_buildings == 5:
            space.current_rent = space.rent_hotel

        if config.verbose['update_building_rent']:
            logger.info('Space {space_name} now rent {rent}'.format(space_name=space.name, rent=space.get_rent()))

    def sell_all_buildings(self):
        for key in self.properties:
            if self.properties[key][0].get_type() == 'Street':
                self.sell_all_monopoly_building(key)

    def go_bankrupt(self, creditor=None):
        if creditor:
            creditor.jail_cards += self.jail_cards
            self.sell_all_buildings()
            creditor.cash += self.cash
            went_bankrupt = False
            for key in self.properties:
                for space in self.properties[key]:
                    if space.monopoly not in creditor.properties:
                        creditor.properties[space.monopoly] = []
                    creditor.properties[space.monopoly].append(space)
                    space.owner = creditor

                    if config.verbose['new_owner']:
                        logger.info('Player {} is the new owner of space {}'.format(creditor.id, space.name))

                    if space.is_mortgaged:
                        creditor.pay_bank_interest(space)
                        if creditor.is_bankrupt:
                            went_bankrupt = True
                            break
                        mask_params = [False, False, False, False]
                        action_mask = creditor.get_action_mask(mask_params)

                        if creditor.can_unmortgage(space):
                            action_mask[space.position_in_action + 1] = 1.
                        action_mask_gpu = torch.FloatTensor(action_mask).to(creditor.device)

                        state = creditor.game.get_state(creditor)
                        with torch.no_grad():
                            value, action, action_log_prob = creditor.policy.act(state, creditor.cash, action_mask_gpu)

                        do_nothing = creditor.apply_action(action)

                        next_state = creditor.game.get_state(creditor)
                        reward = creditor.game.get_reward(creditor, next_state)

                        last_available_action = action_mask_gpu.cpu().argmax().item()
                        if last_available_action != 0:
                            creditor.storage.push(state, action, action_log_prob, value, reward, [1.0])

                    creditor.update_rent(space)
                if went_bankrupt:
                    break
        else:
            self.cash = 0
            if self.game.players_left > 2:
                for key in self.properties:
                    for space in self.properties[key]:
                        space.nullify()
                        self.game.auction(self, space)
        self.cash = 0
        self.jail_cards = 0
        self.properties = {}
        self.is_bankrupt = True
        self.position = 0

        mask_params = [False, False, False, False]
        action_mask = self.get_action_mask(mask_params)
        action_mask_gpu = torch.FloatTensor(action_mask).to(self.device)

        state = self.game.get_state(self)
        with torch.no_grad():
            value, action, action_log_prob = self.policy.act(state, self.cash, action_mask_gpu)

        reward = self.game.get_reward(self, state, result=-1)

        mask = [0.0]

        self.storage.push(state, action, action_log_prob, value, reward, mask)


    def won(self):
        mask_params = [False, False, False, False]
        action_mask = self.get_action_mask(mask_params)
        action_mask_gpu = torch.FloatTensor(action_mask).to(self.device)

        state = self.game.get_state(self)
        with torch.no_grad():
            value, action, action_log_prob = self.policy.act(state, self.cash, action_mask_gpu)

        reward = self.game.get_reward(self, state, result=1)

        mask = [0.0]

        self.storage.push(state, action, action_log_prob, value, reward, mask)

    def draw(self):
        mask_params = [False, False, False, False]
        action_mask = self.get_action_mask(mask_params)
        action_mask_gpu = torch.FloatTensor(action_mask).to(self.device)

        state = self.game.get_state(self)
        with torch.no_grad():
            value, action, action_log_prob = self.policy.act(state, self.cash, action_mask_gpu)

        reward = self.game.get_reward(self, state, result=-1)

        mask = [0.0]

        self.storage.push(state, action, action_log_prob, value, reward, mask)

    def pay_bank_interest(self, space):
        bank_interest = space.price * 0.1
        if self.have_enough_money(bank_interest):
            self.pay(bank_interest)
        else:
            self.try_to_survive()
            if self.have_enough_money(bank_interest):
                self.pay(bank_interest)
            else:
                if config.verbose['cant_pay_bank_interest']:
                    logger.info(
                        'Player {id} could not bank interest for space {name}. Player {id} has {cash} but need {interest}'.format(
                            id=self.id,
                            cash=self.cash,
                            name=space.name,
                            interest=bank_interest))
                self.go_bankrupt()

    def try_to_survive(self):
        self.reset_mortgage_buy()

        if config.verbose['try_to_survive']:
            logger.info('Player {id} tries to survive. Have money {cash}'.format(id=self.id, cash=self.cash))
        while True:
            mask_params = [False, True, False, False]
            action_mask = self.get_action_mask(mask_params)
            action_mask_gpu = torch.FloatTensor(action_mask).to(self.device)

            state = self.game.get_state(self)
            with torch.no_grad():
                value, action, action_log_prob = self.policy.act(state, self.cash, action_mask_gpu)

            do_nothing = self.apply_action(action)

            next_state = self.game.get_state(self)
            reward = self.game.get_reward(self, next_state)

            last_available_action = action_mask_gpu.cpu().argmax().item()
            if last_available_action != 0:
                self.storage.push(state, action, action_log_prob, value, reward, [1.0])

            if do_nothing:
                break


    def compute_total_wealth(self):
        total_wealth = self.cash
        for key in self.properties:
            for space in self.properties[key]:
                if space.get_type() == 'Street':
                    total_wealth += space.n_buildings * space.build_cost
                space_cost = space.price
                if space.is_mortgaged:
                    space_cost /= 2
                total_wealth += space_cost
        self.total_wealth = total_wealth


    def reward_wealth(self):
        money = self.cash * 0.5
        income = 0
        for key in self.properties:
            for space in self.properties[key]:
                if not space.is_mortgaged:
                    income += space.current_rent
        reward = (money + income) / 10000
        return reward

    def get_income(self):
        income = 0
        for key in self.properties:
            for space in self.properties[key]:
                if not space.is_mortgaged:
                    income += space.current_rent
        return income

    def obs_equals(self, elem1, elem2):
        r = torch.all(torch.eq(elem1, elem2))
        return r.item() == 1

    def reward_equals(self, elem1, elem2):
        return elem1.item() == elem2.item()
