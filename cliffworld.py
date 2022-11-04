import copy


def print_world(world, pos):
    """
    Pretty-prints the Cliffworld

    :param world: CliffWorld object to be printed
    :param pos: Position of the agent
    :return: None, prints the grid
    """
    w = copy.deepcopy(world)
    w[pos[0]][pos[1]] = 4
    for i in range(len(w[0])):
        print("__", end='')
    print("_")
    for r in w:
        print("|", end='')
        for c in r:
            if c == 0:
                print("O|", end='')
            elif c == 1:
                print("W|", end='')
            elif c == 3:
                print("E|", end='')
            elif c == 2:
                print("S|", end='')
            elif c == 4:
                print("A|", end='')
        print()
    for i in range(len(w[0])):
        print("__", end='')
    print("_")


class CliffWorld:
    """
    Cliff Walking environment described on page 132 of Sutton and Barto
    """

    def __init__(self):
        self.world = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]]

        self.pos = [3, 0]

    def at_goal(self):
        """
        Returns if the agent is at the goal state

        :return: True if agent is at goal state, false otherwise
        """
        return self.pos[0] == 3 and self.pos[1] == 11

    def off_cliff(self):
        """
        Returns if the agent is off the cliff

        :return: True if the agent is off the cliff, false otherwise
        """
        return self.pos[0] == 3 and 1 <= self.pos[1] <= 10

    def move(self, d_y, d_x):
        """
        Moves the agent by a given amount vertically and horizontally

        :param d_y: Change in vertical position
        :param d_x: Change in horizontal position
        :return: None, changes the position of the agent in the cliff world
        """
        if 0 <= self.pos[0] - d_y <= 3 and 0 <= self.pos[1] + d_x <= 11:
            self.pos[0] -= d_y
            self.pos[1] += d_x
        pass
