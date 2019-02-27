import config
from random import shuffle
import sys
sys.path.append("/monopoly")


n_games = 1000  # games are basically episodes
n_rounds = 100  # and rounds are steps

# TODO: add better logging and more statistics


def main():
    win_stats = {str(i): 0 for i in range(config.n_players)}

    for n_game in range(n_games):

        players = [Player(policy=RandomAgent(), jail_policy=RandomJailPolicy(), player_id=i) for i in range(config.n_players)]
        # players.append(Player(policy=FixedAgent(), player_id=222))
        # shuffle(players)

        game = Game(players=players, dice_class=Dice)

        for n_round in range(n_rounds):

            # TODO: change this, don't like two completely the same conditions
            if not game.is_game_active():     # stopping rounds loop
                break

            game.update_round()

            for player in game.players:

                if not game.is_game_active():  # stopping players loop
                    break

                game.pass_dice()

                while True:
                    if not game.is_game_active():  # stopping players loop
                        break

                    player.before_roll_act(game)

                    game.dice.roll()

                    if player.is_in_jail(): # need to handle situation when he stayed in prison for 3 moves (he must pay)
                        stay_in_jail = player.choose_jail_strategy(dice=game.dice)
                        if stay_in_jail:
                            player.build(space, game)
                            break

                    if game.dice.double_counter == 3:
                        player.go_to_jail()  # in go_to_jail call player.choose_jail_policy so that the player
                        break                              # leave prison right away

                    player.move(game.dice)

                    if player.position == 30:
                        player.go_to_jail() # the same here
                        break

                    # TODO: add card go to jail

                    space = game.board[player.position]

                    player.act(space, game)

                    if player.is_bankrupt:
                        game.remove_player(player)
                        break

                    if game.dice.double:
                        continue

                    # end turn
                    break

        win_stats[str(game.players[0].id)] += 1

        print('Player {} is on the 1 place'.format(game.players[0].id))
        for i in range(len(game.lost_players)):
            print('Player {} is on the {} place '.format(game.lost_players[i].id, i + 2))

    for key in win_stats:
        print('Player {} won {} / {}'.format(key, win_stats[key], n_games))
        print('-------Win rate is {:.2f} %'.format(win_stats[key] / n_games * 100))
























if __name__ == '__main__':
    main()
