import numpy as np
from utils import *


class GridWorld:
    """
    Represents a grid world Markov Decision Process (MDP).
    """
    def __init__(self, dimensions, num_obstacles=0, goal_state=None, correct_action_prob=0.8, gamma=1.0):
        """
        Creates a grid world Markov Decision Process (MDP).

        :param dimensions: dimensions of the grid.
        :type dimensions: bidimensional tuple of ints.
        :param num_obstacles: number of obstacles.
        :type num_obstacles: int.
        :param goal_state: the goal state.
        :type goal_state: bidimensional tuple of ints.
        :param correct_action_prob: probability of correctly executing the chosen action.
        :type correct_action_prob: float.
        :param gamma: discount factor.
        :type gamma: float.
        """
        self.dimensions = dimensions
        self.grid = np.zeros((dimensions[0], dimensions[1]), dtype=int)
        if not goal_state:
            self.grid[dimensions[0] - 1, dimensions[1] - 1] = GOAL
        else:
            self.grid[goal_state[0], goal_state[1]] = GOAL
        self.make_world(num_obstacles)
        self.correct_action_prob = correct_action_prob
        self.gamma = gamma

    def make_world(self, num_obstacles):
        """
        Creates the random obstacles.

        :param num_obstacles: number of obstacles.
        :type num_obstacles: int.
        """
        for o in range(num_obstacles):
            cell = UNDEFINED
            while cell != EMPTY:
                i = np.random.randint(0, self.grid.shape[0])
                j = np.random.randint(0, self.grid.shape[1])
                cell = self.grid[i, j]
                if cell == EMPTY:
                    self.grid[i, j] = OBSTACLE

    def is_cell_valid(self, position):
        """
        Checks if a given cell is valid (within the grid boundaries and no obstacles present).

        :param position: cell position.
        :type position: bidimensional tuple of ints.
        :return: if the given cell is valid.
        :rtype: bool.
        """
        if position[0] < 0 or position[0] >= self.dimensions[0] or position[1] < 0 or position[1] >= self.dimensions[1]:
            return False
        if self.grid[position[0], position[1]] == OBSTACLE:
            return False
        return True

    def count_neighborhood_obstacles(self, position):
        """
        Counts the number of obstacles in the neighborhood of the given position.

        :param position: cell position.
        :type position: bidimensional tuple of ints.
        :return: number of obstacles in the neighborhood of the given position.
        :rtype: int.
        """
        neighbors = [(position[0] - 1, position[1]), (position[0], position[1] - 1), (position[0] + 1, position[1]),
                     (position[0], position[1] + 1)]
        count = 0
        for neighbor in neighbors:
            if not self.is_cell_valid(neighbor):
                count += 1
        return count

    def get_valid_sucessors(self, current_state, action=UNDEFINED):
        """
        Gets valid sucessors of the current state.

        :param current_state: the current state.
        :type current_state: bidimensional tuple of ints.
        :param action: chosen action.
        :type action: int (STOP, UP, RIGHT, DOWN or LEFT).
        :return: list of sucessors.
        :rtype: list of bidimensional tuples of ints.
        """
        candidates = [(current_state[0], current_state[1]), (current_state[0] - 1, current_state[1]),
                     (current_state[0], current_state[1] - 1),
                     (current_state[0] + 1, current_state[1]),
                     (current_state[0], current_state[1] + 1)]
        valid_sucessors = []
        for candidate in candidates:
            if self.is_cell_valid(candidate):
                valid_sucessors.append(candidate)
        return valid_sucessors

    def get_cell_value(self, position):
        """
        Gets the value of the given cell.

        :param position: position of the given cell.
        :rtype position: bidimensional tuple of ints.
        :return: value of the given cell.
        :type: int (EMPTY, GOAL or OBSTACLE).
        """
        return self.grid[position[0], position[1]]

    def predict_next_state_given_action(self, current_state, action):
        """
        Predicts the most probable next state given the current state and chosen action.

        :param current_state: the current state.
        :type current_state: bidimensional tuple of ints.
        :param action: chosen action.
        :type action: int (STOP, UP, RIGHT, DOWN or LEFT).
        :return: most probable next state.
        :rtype: bidimensional tuple of ints.
        """
        if action == UP:
            next_state = current_state[0] - 1, current_state[1]
        elif action == RIGHT:
            next_state = current_state[0], current_state[1] + 1
        elif action == DOWN:
            next_state = current_state[0] + 1, current_state[1]
        elif action == LEFT:
            next_state = current_state[0], current_state[1] - 1
        else:
            next_state = current_state
        return next_state

    def transition_probability(self, current_state, action, next_state):
        """
        Computes the transition probability given the current state, the chosen action, and the next state, i.e.
        p(s,a,s').

        :param current_state: the current state.
        :type current_state: bidimensional tuple of ints.
        :param action: chosen action.
        :type action: int (STOP, UP, RIGHT, DOWN or LEFT).
        :param next_state: the next state.
        :type next_state: bidimensional tuple of ints.
        :return: transition probability p(s,a,s').
        :rtype: float.
        """
        if (not self.is_cell_valid(current_state)) or (not self.is_cell_valid(next_state)):
            return 0.0
        di = next_state[0] - current_state[0]
        dj = next_state[1] - current_state[1]
        if abs(di) > 1 or abs(dj) > 1:
            return 0.0
        if abs(di) != 0 and abs(dj) != 0:
            return 0.0
        if action == STOP:
            if di == 0 and dj == 0:
                return 1.0
            else:
                return 0.0
        predicted_next_state = self.predict_next_state_given_action(current_state, action)
        if next_state == predicted_next_state:
            return self.correct_action_prob
        count = self.count_neighborhood_obstacles(current_state)
        mistake_prob = (1.0 - self.correct_action_prob) / (NUM_ACTIONS - 1)
        if di == 0 and dj == 0:
            no_move_prob = mistake_prob + count * mistake_prob
            if not self.is_cell_valid(predicted_next_state):
                no_move_prob += self.correct_action_prob - mistake_prob
            return no_move_prob
        return mistake_prob

    def reward(self, current_state, action):
        """
        Computes the expected reward given the current state and action.

        :param current_state: the current state.
        :type current_state: bidimensional tuple of ints.
        :param action: chosen action.
        :type action: int (STOP, UP, RIGHT, DOWN or LEFT).
        :return: expected reward.
        :rtype: float.
        """
        if self.grid[current_state[0], current_state[1]] == GOAL:
            return 0.0
        return -1.0
