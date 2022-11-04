from cliffworld import CliffWorld
from cliffworld import print_world


def human_input():
    """
    User interface loop for the cliff walking example on page 132
    of Sutton and Barto

    :return: None
    """
    print("Initializing Cliff World...")
    c = CliffWorld()
    print_world(c.world, c.pos)
    print("Ready to begin? [y/n]:")
    start = input()
    while start != 'q' and start != 'n':
        print("Do you want to move up, down, left, or right? ['U', 'D', 'L', 'R']")
        dir = input()
        actions = {'U': [1, 0], 'D': [-1, 0], 'L': [0, -1], 'R': [0, 1]}
        A = None
        if dir in actions:
            A = actions[dir]
            if not c.move(A[0], A[1]):
                print("You fell off the cliff! Just like another sheep...")
                break
            print_world(c.world, c.pos)
            if c.at_goal():
                print("Reached the end! Woo-Hoo!")
                break
        else:
            print("Please select a valid action! ['U', 'D', 'L', 'R']")

        print("Want to make another move? [y/n]:")
        start = input()


if __name__ == '__main__':
    human_input()
