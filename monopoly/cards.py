class Card(object):

    def __init__(self, attrib):
        self.name = attrib['name']            # Property name
        self.type = attrib['class']           # Class

    def get_type(self):
        return self.type

class MoveTo(Card):

    def __init__(self, attrib):
        Card.__init__(self, attrib)

        self.to_position = attrib['position']


    def apply(self, player, game):
        if player.position < self.to_position:
            player.cash += 200

        player.position = self.to_position

class MoveToUnitily(Card):
    # Move to nearest Unitily
    def __init__(self, attrib):
        Card.__init__(self, attrib)

    def apply(self, player, game):
        player_position = player.position
        while True:
            if player_position >= 40:
                player_position -= 40
                player.cash += 200

            if game.board[player_position].get_type() == 'Unitily':
                player.position = player_position
                break

            player_position += 1



class MoveToRailroad(Card):
    # Move to nearest Railroad
    def __init__(self, attrib):
        Card.__init__(self, attrib)


    def apply(self, player, game):
        player_position = player.position
        while True:
            if player_position >= 40:
                player_position -= 40
                player.cash += 200

            if game.board[player_position].get_type() == 'Railroad':
                player.position = player_position
                break

            player_position += 1

class GetFromBank(Card):

    def __init__(self, attrib):
        Card.__init__(self, attrib)

        self.money = attrib['money']


    def apply(self, player, game):
        player.cash += self.money

class Repairs(Card):

    def __init__(self, attrib):
        Card.__init__(self, attrib)

        self.per_house = attrib['per_house']
        self.per_hotel = attrib['per_hotel']


    def apply(self, player, game):
        player.cash -= player.count_houses() * self.per_house + player.count_hotels() * self.per_hotel


class PayBank(Card, player, game):

    def __init__(self, attrib):
        Card.__init__(self, attrib)

        self.price = attrib['price']


    def apply(self, player, game):
        player.cash -= self.price

class MoveBack(Card):

    def __init__(self, attrib):
        Card.__init__(self, attrib)

        self.n_spaces = attrib['n_spaces']


    def apply(self, player, game):
        player.position -= self.n_spaces

class PayEachPlayer(Card):

    def __init__(self, attrib):
        Card.__init__(self, attrib)

        self.price = attrib['price']


    def apply(self, player, game):
        player.cash -= self.price * (game.players_left - 1)
        for game_player in game.players:
            if player.id != game_player.id:
                game_player.cash += self.price


class GetFromEachPlayer(Card):

    def __init__(self, attrib):
        Card.__init__(self, attrib)

        self.money = attrib['money']

    def apply(self, player, game):
        player.cash += self.money * (game.players_left - 1)
        for game_player in game.players:
            if player.id != game_player.id:
                game_player.cash -= self.money

class GoToJail(Card):

    def __init__(self, attrib):
        Card.__init__(self, attrib)


    def apply(self, player, game):
        player.go_to_jail()


class GetOutOfJail(Card):

    def __init__(self, attrib):
        Card.__init__(self, attrib)


    def apply(self, player, game):
        player.jail_turns = 0
