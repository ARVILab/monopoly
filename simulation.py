from random import shuffle
import sys
import logging

import torch

from policies.random import RandomAgent
from policies.fixed import FixedAgent
from monopoly.player import Player
from monopoly import config
from monopoly.game import Game

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


n_games = 1   # games are basically episodes
n_rounds = 2  # and rounds are steps

# TODO: add better logging and more statistics

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # win_stats = {str(i): 0 for i in range(config.n_players)}
    # win_stats['000'] = 0
    # win_stats['111'] = 0
    # win_stats['222'] = 0
    win_stats = {'000': 0, '111': 0}
    # win_stats['0'] = 0
    full_games_counter = 0

    for n_game in range(n_games):

        if config.verbose['game_start']:
            logger.info('----------------STARTING GAME {}----------------\n'.format(n_game))

        # players = [Player(policy=RandomAgent(), player_id=i) for i in range(config.n_players)]
        players = []
        # players.append(Player(policy=FixedAgent(high=500, low=200, jail=50), player_id='000'))
        players.append(Player(policy=FixedAgent(high=400, low=200, jail=100), player_id='000', device=device))
        players.append(Player(policy=FixedAgent(high=350, low=150, jail=100), player_id='111', device=device))
        # shuffle(players)

        game = Game(players=players)


        for player in players:
            player.set_game(game)

        storage.add_obs(game.get_state(players[0]), step=0)
        storage.to(device)

        for n_round in range(n_rounds):

            # TODO: change this, don't like three completely the same conditional statements
            if not game.is_game_active():     # stopping rounds loop
                break

            game.update_round()

            for player in game.players:
                print('\n\n\n-----------PLAYER {}-----------'.format(player.index))

                if player.is_bankrupt:            # must change it. do it two times because some players can go bankrupt when must pay bank interest
                    game.remove_player(player)    # other player's mortgaged spaces
                    break

                if not game.is_game_active():  # stopping players loop
                    break

                game.pass_dice()

                while True:
                    player.show()
                    if not game.is_game_active():  # stopping players loop
                        break

                    player.optional_actions()

                    game.dice.roll()

                    if player.is_in_jail():
                        stay_in_jail = player.jail_strategy(dice=game.dice)
                        if stay_in_jail:
                            player.optional_actions()
                            break

                    if game.dice.double_counter == 3:
                        player.go_to_jail()
                        break

                    player.move(game.dice.roll_sum)


                    if player.position == 30:
                        player.go_to_jail() # the same here
                        break

                    # TODO: add card go to jail

                    space = game.board[player.position]

                    player.act(space)

                    if player.is_bankrupt:
                        game.remove_player(player)
                        break

                    if game.dice.double:
                        continue

                    player.show()
                    state = game.get_state(player)
                    print('----------STATE---------')
                    print('player_makes', 'opponents_make', 'player_owns', 'opponents_own')
                    counter = 0
                    for i in range(0, len(state) - 3, 4):
                        print('{:12} {}'.format(config.monopolies[counter], state[i:i + 4]))
                        counter += 1
                    print('Position:', state[40])
                    print('Properties value:', state[41])
                    print('Money:', state[42])
                    print('----------REWARD---------')
                    reward = game.get_reward(player, state)
                    print('Reward:', reward)
                    # end turn
                    break


        if game.players_left != 1:
            leaderboard = game.get_leaderboard()
            win_stats[str(leaderboard[0].id)] += 1
            for i, player in enumerate(leaderboard):
                print('Player {} is on the {} place. Total wealth {}'.format(player.id, i + 1, player.total_wealth))
                player.show()
            if len(game.lost_players) != 0:
                for i, player in enumerate(game.lost_players):
                    print('Player {} is on the {} place'.format(player.id, i + 1 + len(leaderboard)))
                    player.show()
        else:
            win_stats[str(game.players[0].id)] += 1
            full_games_counter += 1

            print('Player {} is on the 1 place'.format(game.players[0].id))
            game.players[0].show()
            for i, player in enumerate(game.lost_players):
                print('Player {} is on the {} place '.format(player.id, i + 2))
                player.show()

        players[0].storage.show()

    for key in win_stats:
        print('Player {} won {} / {}'.format(key, win_stats[key], n_games))
        print('-------Win rate is {:.2f} %'.format(win_stats[key] / n_games * 100))

    print('Full games played', full_games_counter)
























if __name__ == '__main__':
    main()
