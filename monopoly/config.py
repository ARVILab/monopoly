n_players = 2                  # Number of players

board_filename = 'board.csv'              # Board filename
chance_cards_filename = 'chance_cards.csv' # chance cards filename
chest_cards_filename = 'chest_cards.csv'   # community chest cards filename

verbose = {'move': True,       # logging verbose
           'pay': True,
           'buy': True,
           'round': True,
           'dice': True,
           'game_start': True,
           'is_game_active': False,
           'optional_actions': True}



monopolies = ('Brown', 'Light Blue', 'Pink',
            'Orange', 'Red', 'Yellow', 'Green', 'Dark Blue', 'Railroad', 'Utility') # monopoly order for state


# action_space = 115
