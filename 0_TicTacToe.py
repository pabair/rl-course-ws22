

game_state = ["_" for i in range(0, 9)]


def print_board():
    print(game_state[0:3])
    print(game_state[3:6])
    print(game_state[6:9])


def update_board(position, symbol):
    # Task: Check if the position is valid (0-8)
    # Task: Check if the position is empty
    # otherwise return with a invalid message to the user
    game_state[position] = symbol


def game_finished():
    # Write logic to decide if game is finished
    # and print the result ("win", "loose", "draw")
    return False


if __name__ == "__main__":
    print("Welcome to TicTacToe!")
    print("You can put your 'x' at the following positions:")
    print('[0,1,2]\n[3,4,5]\n[6,7,8]')

    print("Current board:")
    print_board()
    while not game_finished():
        i = int(input("Where do you want to put your 'x'? (0-8)"))
        update_board(i, "x")
        # Task: implement the opponents move
        print_board()
