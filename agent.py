import numpy as np
import matplotlib.pyplot as plt
from cliffworld import CliffWorld
import copy


class Agent:
    """
    Reinforcement learning agent for the cliff walking example from Sutton and
    Barto. Contains implementations of Sarsa, Q-learning, and on-policy,
    first visit MC Control.
    """

    def __init__(self, alpha, epsilon, gamma):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.q = None
        self.returns = None
        self.n = None
        self.c = None
        self.actions = ['U', 'D', 'L', 'R']
        self.action_to_delta = {'U': [1, 0], 'D': [-1, 0], 'L': [0, -1], 'R': [0, 1]}
        self.ep_reward = 0

    def generate_max_action(self, pos):
        """
        Generates the greedy action using the agent's current Q-table and
        the position as input.

        :param pos: Position of the agent.
        :return: The greedy action to take, one of ['U', 'D', 'L', 'R']
        """
        (y, x) = pos
        q_values = [self.q[((y, x), a)] for a in self.actions]
        m = max(q_values)
        max_indices = [i for i, x in enumerate(q_values) if x == m]
        A = self.actions[np.random.choice(max_indices)]
        return A

    def generate_egreedy_action(self, pos):
        """
        Generates the e-greedy action using the agent's current Q-table and
        the position as input. Selects the greedy action with probability
        1 - e and a random action with probability e.

        :param pos: Position of the agent.
        :return: The e-greedy action to take, one of ['U', 'D', 'L', 'R']
        """
        (y, x) = pos
        q_values = [self.q[((y, x), a)] for a in self.actions]
        random_val = np.random.rand()
        if random_val > self.epsilon:
            m = max(q_values)
            max_indices = [i for i, x in enumerate(q_values) if x == m]
            A = self.actions[np.random.choice(max_indices)]
        else:
            A = np.random.choice(self.actions)

        return A

    def QLearning(self, episodes, rewards, runs, anneal):
        """
        Q-learning implementation, pseudocode found on page 131 of Sutton
        and Barto.

        :param episodes: Number of episodes per independent run
        :param rewards: Empty reward table for displaying results
        :param runs: Number of independent runs
        :param anneal: Boolean indicating whether to use gradually
        decreasing epsilon values
        :return: None, modifies the agent's Q-table
        """
        for r in range(runs):
            print(f"Run: {r + 1}")
            self.q = {((y, x), a): 0
                      for y in range(4)
                      for x in range(12)
                      for a in ['U', 'D', 'L', 'R']}

            for i in range(episodes):
                if anneal:
                    self.epsilon = 1 / (i + 1)
                curr = CliffWorld()
                self.ep_reward = 0
                np.random.seed((i + 1) * (r + 1))

                while not curr.off_cliff() and not curr.at_goal():
                    y, x = curr.pos
                    A = self.generate_egreedy_action((y, x))

                    delta_y, delta_x = self.action_to_delta[A]

                    curr.move(delta_y, delta_x)

                    new_y, new_x = curr.pos

                    if curr.off_cliff():
                        R = -100

                    else:
                        R = -1

                    next_A = self.generate_max_action((new_y, new_x))
                    self.q[(y, x), A] = self.q[(y, x), A] + self.alpha * \
                                        (R + (self.gamma * self.q[(new_y, new_x), next_A]) - self.q[(y, x), A])

                    self.ep_reward += R

                rewards[i] += self.ep_reward

    def Sarsa(self, episodes, rewards, runs, anneal):
        """
        Sarsa implementation, pseudocode found on page 130 of Sutton
        and Barto.

        :param episodes: Number of episodes per independent run
        :param rewards: Empty reward table for displaying results
        :param runs: Number of independent runs
        :param anneal: Boolean indicating whether to use gradually
        decreasing epsilon values
        :return: None, modifies the agent's Q-table
        """
        for r in range(runs):
            print(f"Run: {r + 1}")
            self.q = {((y, x), a): 0
                      for y in range(4)
                      for x in range(12)
                      for a in ['U', 'D', 'L', 'R']}

            for i in range(episodes):
                if anneal:
                    self.epsilon = 1 / (i + 1)
                self.ep_reward = 0
                curr = CliffWorld()
                np.random.seed((i + 1) * (r + 1))

                # Initialize S
                y, x = curr.pos

                # Choose A from S using e-greedy policy
                A = self.generate_egreedy_action((y, x))

                while not curr.at_goal() and not curr.off_cliff():
                    delta_y, delta_x = self.action_to_delta[A]

                    # Take action A, observe R, S'
                    curr.move(delta_y, delta_x)

                    new_y, new_x = curr.pos

                    if curr.off_cliff():
                        R = -100

                    else:
                        R = -1

                    next_A = self.generate_egreedy_action((new_y, new_x))

                    self.q[(y, x), A] = self.q[(y, x), A] + self.alpha * \
                                        (R + (self.gamma * self.q[(new_y, new_x), next_A]) - self.q[(y, x), A])

                    # Update state S and A with S' and A'
                    y, x = new_y, new_x
                    A = next_A

                    self.ep_reward += R

                rewards[i] += self.ep_reward

    def print_policy(self, cliff):
        """
        Print the current greedy policy overlaid on the cliff world.

        :param cliff: The CliffWorld object to print.
        :return: None, prints the policy grid.
        """
        w = copy.deepcopy(cliff.world)
        for i in range(len(w[0])):
            print("__", end='')
        print("_")
        for i in range(len(w)):
            print("|", end='')
            for j in range(len(w[0])):
                if w[i][j] == 0:
                    print("O|", end='')
                else:
                    q_values = [self.q[((i, j), a)] for a in self.actions]
                    A = self.actions[np.argmax(q_values)]
                    print(A + '|', end='')
            print()
        for i in range(len(w[0])):
            print("__", end='')
        print("_")


# Driver loop of the simulator
def main():
    print("Welcome to the TD Learning Cliff Walking simulator!")
    print("Would you like to use gradually decreasing epsilon values? [y/n]")
    yes = input()
    if yes == 'y':
        anneal = True
        epsilon = 0
    else:
        anneal = False
        print("Would you like to select the agent's epsilon value? [y/n] Default" +
              " value is 0.1")
        yes = input()
        if yes == 'y':
            print("What would you like the agent's epsilon value to be?")
            epsilon = float(input())
        else:
            epsilon = 0.1
    print("Would you like to select the agent's alpha value? [y/n] Default" +
          " value is 0.1")
    yes = input()
    if yes == 'y':
        print("What would you like the agent's alpha value to be?")
        alpha = float(input())
    else:
        alpha = 0.1
    print("Would you like to select the agent's gamma value? [y/n] Default" +
          " value is 0.95")
    yes = input()
    if yes == 'y':
        print("What would you like the agent's gamma value to be?")
        gamma = float(input())
    else:
        gamma = 0.95
    print("Now simulating 500 independent runs on the Cliff Walking World " +
          "with Sarsa and Q-learning! Each run" +
          " contains 500 episodes!")

    a = Agent(alpha, epsilon, gamma)

    print("Running Q-learning...")
    q_rewards = np.zeros(500)
    a.QLearning(500, q_rewards, 500, anneal)
    print("Q-Learning policy map:")
    a.print_policy(CliffWorld())

    print("Running Sarsa...")
    sarsa_rewards = np.zeros(500)
    a.Sarsa(500, sarsa_rewards, 500, anneal)
    print("Sarsa's policy map:")
    a.print_policy(CliffWorld())

    figure, axes = plt.subplots(1, 1)
    axes.plot(q_rewards / 500, label='Q-Learning')
    axes.plot(sarsa_rewards / 500, label='Sarsa')
    axes.legend()
    axes.set_xlabel('Episodes')
    axes.set_ylabel('Sum of rewards during episode')
    plt.show()


if __name__ == '__main__':
    main()
