n_players = 3                  # Number of players

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
           'optional_actions': True,
           'auction': True,
           'auction_process': True,
           'mortgage': True,
           'unmortgage': True,
           'go': True,
           'try_to_survive': True,
           'update_building_rent': True,
           'buy_building': True,
           'new_owner': True,
           '3_moves_in_jail': True,
           'stay_in_jail': True,
           'double_leave_jail': True,
           'pay_to_leave_jail': True,
           'sell_building': True,
           'on_mortgaged_property': True}



monopolies = ('Brown', 'Light Blue', 'Pink',
            'Orange', 'Red', 'Yellow', 'Green', 'Dark Blue', 'Railroad', 'Utility') # monopoly order for state


action_space = 60
state_space = 43
