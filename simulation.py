from random import shuffle
import sys
import logging

from policies.random import RandomAgent
from monopoly.dice import Dice
from monopoly.player import Player
from monopoly import config
from monopoly.game import Game

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


n_games = 1   # games are basically episodes
n_rounds = 10  # and rounds are steps

# TODO: add better logging and more statistics


def main():
    win_stats = {str(i): 0 for i in range(config.n_players)}

    for n_game in range(n_games):

        if config.verbose['game_start']:
            logger.info('----------------STARTING GAME {}----------------\n'.format(n_game))

        players = [Player(policy=RandomAgent(), player_id=i) for i in range(config.n_players)]
        # players.append(Player(policy=FixedAgent(), player_id=222))
        # shuffle(players)

        game = Game(players=players)

        for player in players:
            player.set_game(game)

        for n_round in range(n_rounds):

            # TODO: change this, don't like three completely the same conditional statements
            if not game.is_game_active():     # stopping rounds loop
                break

            game.update_round()

            for player in game.players:

                if player.is_bankrupt:            # must change it. do it two times because some players can go bankrupt when must pay bank interest
                    game.remove_player(player)    # other player's mortgaged spaces
                    break

                if not game.is_game_active():  # stopping players loop
                    break

                game.pass_dice()

                while True:
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

                    # end turn
                    break

        # if game.players_left != 1:
            # get the richest player. he will be the winner
            # pass
        # else:
        win_stats[str(game.players[0].id)] += 1

        print('Player {} is on the 1 place'.format(game.players[0].id))
        for i in range(len(game.lost_players)):
            print('Player {} is on the {} place '.format(game.lost_players[i].id, i + 2))

    for key in win_stats:
        print('Player {} won {} / {}'.format(key, win_stats[key], n_games))
        print('-------Win rate is {:.2f} %'.format(win_stats[key] / n_games * 100))
























if __name__ == '__main__':
    main()
