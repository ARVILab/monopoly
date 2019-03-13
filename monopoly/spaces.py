from . import dice
from random import shuffle

class Space:
    """Generic space object that initializes the two attributes shared by all spaces: Name and position on the board."""

    def __init__(self, attrib):

        self.name = attrib['name']            # Property name
        self.position = attrib['position']    # Board position
        self.type = attrib['class']           # Class
        self.position_in_action = attrib['position_in_action']          # index of property in action

    def get_type(self):
        return self.type

class Property(Space):
    """Generic property object with the attributes shared by the three space types that can be owned: Streets,
    railroads, and utilities. Inherits attributes from the Space object."""

    def __init__(self, attrib):

        Space.__init__(self, attrib)

        self.monopoly = attrib['monopoly']                              # Name of monopoly
        self.monopoly_size = attrib['monopoly_size']                    # Number of properties in monopoly
        self.price = attrib['price']                                    # Price to buy
        self.price_mortgage = self.price / 2                            # Mortgage price
        self.init_rent = attrib['rent']                                 # Initial rent
        self.current_rent = self.init_rent                              # Current rent
        self.monopoly_max_income = attrib['monopoly_max_income']        # maximum money u can make from this monopoly
        self.monopoly_max_price = attrib['monopoly_max_price']          # maximum money u can spend on this monopoly
        self.is_mortgaged = False                                       # Mortgage status
        self.owner = None                                               # Property owner


    def nullify(self):
        self.current_rent = self.init_rent
        self.owner = None
        self.is_mortgaged = False


class Street(Property):
    """Street object that includes attributes related to buildings: cost to build, rent prices at each level of building
    development, and the number of buildings built. Inherits attributes from the Property object."""

    def __init__(self, attrib):

        Property.__init__(self, attrib)

        self.build_cost = attrib['build_cost']        # Building cost
        self.monopoly_rent = self.init_rent * 2       # Rent with monopoly
        self.rent_house_1 = attrib['rent_house_1']    # Rent with 1 house
        self.rent_house_2 = attrib['rent_house_2']    # Rent with 2 houses
        self.rent_house_3 = attrib['rent_house_3']    # Rent with 3 houses
        self.rent_house_4 = attrib['rent_house_4']    # Rent with 4 houses
        self.rent_hotel = attrib['rent_hotel']        # Rent with hotel
        self.n_buildings = 0

        # self.n_houses = 0                             # Number of houses
        # self.n_hotels = 0                             # Number of hotels

    def get_rent(self, roll=None):
        return self.current_rent

    def nullify(self):
        super().nullify()
        self.n_buildings = 0


class Railroad(Property):
    """Railroad object that includes attributes related to rent prices per number of railroads owned. Inherits
    attributes from the Property object."""

    def __init__(self, attrib):

        Property.__init__(self, attrib)

        self.rent_railroad_2 = self.init_rent * 2  # Rent with 2 railroads
        self.rent_railroad_3 = self.init_rent * 4  # Rent with 3 railroads
        self.monopoly_rent = self.init_rent * 8    # Rent with monopoly

    def get_rent(self, roll=None):

        return self.current_rent


class Utility(Property):
    """Utility object that includes attributes related to rent prices in the Utility monopoly. For this monopoly, rents
    are multipliers of dice rolls rather than absolute values. Inherits attributes from the Property object."""

    def __init__(self, attrib):

        Property.__init__(self, attrib)

        self.monopoly_rent = self.init_rent + 6

    def get_rent(self, roll=1):
        return self.current_rent * roll


class Tax(Space):
    """Tax object that lists the tax to be paid by a player that lands on a taxed space. Inherits attributes from the
    Space object."""

    def __init__(self, attrib):

        Space.__init__(self, attrib)

        self.tax = attrib['tax']

class Idle(Space):
    def __init__(self, attrib):
        Space.__init__(self, attrib)

class Jail(Space):
    def __init__(self, attrib):
        Space.__init__(self, attrib)

class CardHolder(Space):

    def __init__(self, attrib, cards):
        Space.__init__(self, attrib)

        # shuffle(cards)
        #
        # self.deck = Queue(cards)

    def take_card(self):
        card = self.deck.pop()

    def put_card(self, card):
        self.deck.add(card)

class Chance(CardHolder):

    def __init__(self, attrib, cards_file='chance.csv'):
        # read cards
        CardHolder.__init__(self, attrib, [])


class Chest(CardHolder):

    def __init__(self, attrib, cards_file='community_chest.csv'):
        # read cards
        CardHolder.__init__(self, attrib, [])


class Queue(object):
    def __init__(self, values):
        self.queue = list(values)

    def add(self, value):
        self.queue.append(value)

    def pop(self):
        value = self.queue[0]
        del self.queue[0]
        return value

    def show(self):
        print(self.queue)
