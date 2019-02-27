import config

class Player:

    def __init__(self, policy=None, jail_policy=None, player_id=None):

        self.id = player_id             # Identification number
        self.index = 0
        self.cash = 1500                # Cash on hand
        self.properties = {}            # Dict of monopolies
        self.position = 0               # Board position
        self.jail_cards = 0             # Number of "Get Out Of Jail Free" cards
        self.jail_turns = 0             # Number of remaining turns in jail
        self.bankrupt = False           # Bankrupt status

        self.policy = policy
        self.jail_policy = jail_policy
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

    def act(self, space, game):
        space_type = space.get_type()
        obligation = self.oligatory_acts[space_type]

        if obligation:
            obligation(game, space, game.dice)

        if not self.is_bankrupt:
            self.optional_actions(space, game)


    def move(self, dice):
        old_position = self.position
        self.position += dice.roll

        if self.position >= 40:
            self.position -= 40
            self.cash += 200

        if config.verbose['move']:
            logger.info('Player {id} moved on board from position {old_position} to {new_position}'.format(
                id=self.id, old_position=old_position, new_position=self.position))

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

    def sell_all_monopoly_building(self, monopoly):
        for space in monopoly:
            self.cash += space.n_buildings * space.build_cost / 2

    def is_in_jail(self):
        if self.jail_turns == 0:
            return False
        elif self.jail_turns > 1:
            self.jail_turns -= 1
            return True

    def go_to_jail(self):
        self.position = 10
        self.jail_turns = 3

        logger.info('Player {id} went to jail'.format(id=self.id))

        self.choose_jail_strategy()


    def before_roll_act(self, game):
        while True:
            mask_params = [True, True, False, False]
            action_mask = self.get_action_mask(game, mask_params)
            state = game.get_state(self)
            action = self.policy.act(state)
            action = self.apply_mask(action, action_mask)
            stop = self.apply_action(action)
            if stop:
                break


    def unowned_street(self, game, space):
        mask_params = [False, False, False, False]
        action_mask = self.get_action_mask(game, mask_params)
        if self.have_enough_money(space.price):
            action_mask[1 + space.position_in_action] = 1.
        state = game.get_state(self)
        action = self.policy.act(state) # don't forget to apply mask
        action = self.apply_mask(action, action_mask)
        stop = self.apply_action(action)
        if stop:
            game.auction(self, space)

        # auction here if not buying street

    def jail_strategy(self, dice=None):
        pass

    def optional_actions(self, space, game):
        pass


    def apply_mask(self, action, mask):
        return np.array(action) * np.array(mask)

    def can_build_on_monopoly(self, monopoly):
        if len(monopoly) != monopoly[0].monopoly_size:
            return False
        return True

    def can_build_on_space(self, space, monopoly):
        if space.n_buildings < 5:
            buildings_list = [s.n_buildings for s in monopoly]
            max_buildings = np.max(buildings_list)
            all_equal = buildings_list[1:] == buildings_list[:-1]
            if space.n_buildings != max_buildings or space.n_buildings == 0 or all_equal:
                if self.have_enough_money(space.build_cost):
                    return True
        return False

    def can_sell_building(self, space, monopoly):
        if space.n_buildings > 0:
            buildings_list = [s.n_buildings for s in monopoly]
            max_buildings = np.max(buildings_list)
            if space.n_buildings == max_buildings:
                return True
        return False

    def can_mortgage(self, space, monopoly):
        buildings_list = [s.n_buildings for s in monopoly]
        max_buildings = np.max(buildings_list)
        if max_buildings == 0 and not space.is_mortgaged:
            return True
        return False

    def can_unmortgage(self, space):
        if space.is_mortgage:
            if self.have_enough_money(space.price_mortgage + space.price_mortgage * 0.1):
                return True
        return False


    def get_action_mask(self, game, params):
        mask = [1.]                                             # do nothing is always available
        spend_money_mask = self.get_spend_money_mask(params[0]) # mask for buying streets and building on those streets
        make_money_mask = self.get_make_money_mask(params[1])   # mask for mortgagin streets and selling building from those streets
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
                if self.can_build_on_monopoly(monopoly):
                    for space in monopoly:
                        if self.can_build_on_space(space, monopoly):
                            mask[space.position_in_action] = 1.
                for space in monopoly:
                    if self.can_unmortgage(space):
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
                    if self.can_mortgage(space, monopoly):
                        mask[space.position_in_action] = 1.
        return mask

    def get_jail_mask(self, is_avaliable):
        mask = np.zeros(2)
        if is_avaliable:
            if self.have_enough_money(50)
                mask[0] = 1.
            if self.jail_cards > 0:
                mask[1] = 1.
        return mask

    def get_trading_mask(self, is_avaliable):
        mask = np.zeros(1)
        if is_avaliable:
            mask[0] = 1.
        return mask

    def act_on_property(self, game, space, dice):
        if space.owner:
            money_owned = space.get_rent(dice.roll_sum)
            if self.have_enough_money(money_owned):
                self.pay(money_owned, space.owner)
            else:
                self.try_to_survive(space)
                if self.have_enough_money(money_owned):
                    self.pay(money_owned, space.owner)
                else:
                    self.go_bankrupt(game, space.owner)
        else:
            self.unowned_street(game, space)


    def act_on_chance(self, space):
        pass

    def act_on_chest(self, space):
        pass

    def act_on_tax(self, game, space):
        money_owned = space.tax
        if self.have_enough_money(money_owned):
            self.pay(money_owned)
        else:
            self.try_to_survive(space)
            if self.have_enough_money(money_owned):
                self.pay(money_owned)
            else:
                self.go_bankrupt(game)

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
                space.current_rent = space.rent
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
                space.current_rent = space.rent
            if diff == 2:
                for s in monopoly:
                    s.current_rent = space.monopoly_rent

    def sell_all_buildings(self):
        for monopoly in self.properties:
            if monopoly[0].get_type() == 'Street':
                self.sell_all_monopoly_building(monopoly)

    def go_bankrupt(self, game, creditor=None):
        if creditor:
            creditor.jail_cards += self.jail_cards
            self.sell_all_buildings()
            creditor.cash += self.cash
            for monopoly in self.properties:
                for space in monopoly:
                    if space.monopoly not in creditor.properties:
                        creditor.properties[space.monopoly] = []
                    creditor.properties[space.monopoly].append(space)
                    space.owner = creditor
                    creditor.update_rent(space)

                    # mortgage 10%
                    money_owned = space.
                    if creditor.have_enough_money(money_owned):
                        creditor.pay(money_owned)
                    else:
                        creditor.try_to_survive(space)
                        if creditor.have_enough_money(money_owned):
                            creditor.pay(money_owned)
                        else:
                            creditor.go_bankrupt(game)
                            break

        else:
            for key in self.properties:
                for space in self.properties[key]:
                    space.nullify()
                    game.auction(self, space)
        self.cash = 0
        self.jail_cards = 0
        self.properties = {}
        self.bankrupt = 0

    def try_to_survive(self, space):
        while True:
            mask_params = [False, True, False, False]
            action_mask = self.get_action_mask(game, mask_params)
            state = game.get_state(self)
            action = self.policy.act(state)
            action = self.apply_mask(action, action_mask)
            stop = self.apply_action(action)
            if stop:
                break
