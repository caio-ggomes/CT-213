import numpy as np
from grid_world import GridWorld
from dynamic_programming import random_policy, greedy_policy, policy_evaluation, policy_iteration, value_iteration
from utils import *


def print_value(value):
    """
    Prints a value function.

    :param value: table (i, j) representing the value function.
    :type value: bidimensional NumPy array.
    """
    print('Value function:')
    dimensions = value.shape
    for i in range(dimensions[0]):
        print('[', end='')
        for j in range(dimensions[1]):
            state = (i, j)
            if grid_world.is_cell_valid(state):
                print('%9.2f' % value[i, j], end='')
            else:
                print('    *    ', end='')
            if j < dimensions[1] - 1:
                print(',', end='')
        print(']')


def print_policy(policy):
    """
    Prints a policy. For a given state, assumes that the policy executes a single action
    or a set of actions with the same probability.

    :param policy: table (i, j, a) representing the policy.
    :type policy: tridimensional NumPy array.
    """
    print('Policy:')
    action_chars = 'SURDL'
    dimensions = policy.shape[0:2]
    for i in range(dimensions[0]):
        print('[', end='')
        for j in range(dimensions[1]):
            state = (i, j)
            if grid_world.is_cell_valid(state):
                cell_text = ''
                for action in range(NUM_ACTIONS):
                    if policy[i, j, action] > 1.0e-3:
                        cell_text += action_chars[action]
            else:
                cell_text = '*'
            cell_text = cell_text.center(9)
            print(cell_text, end='')
            if j < dimensions[1] - 1:
                print(',', end='')
        print(']')


# CORRECT_ACTION_PROB = 1.0  # probability of correctly executing the chosen action
# GAMMA = 1.0  # discount factor
CORRECT_ACTION_PROB = 0.8  # probability of correctly executing the chosen action
GAMMA = 0.98  # discount factor

np.random.seed(0)

dimensions = (6, 6)
num_obstacles = 6
goal_state = (5, 5)

# Instantiating the grid world
grid_world = GridWorld(dimensions, num_obstacles, goal_state, CORRECT_ACTION_PROB, GAMMA)

# Testing policy evaluation
print('Evaluating random policy, except for the goal state, where policy always executes stop:')
policy = random_policy(grid_world)
policy[goal_state[0], goal_state[1], STOP] = 1.0
policy[goal_state[0], goal_state[1], UP:NUM_ACTIONS] = np.zeros(NUM_ACTIONS - 1)
initial_value = np.zeros(dimensions)
value = policy_evaluation(grid_world, initial_value, policy)
print_value(value)
print_policy(policy)
print('----------------------------------------------------------------\n')

# Testing value iteration
print('Value iteration:')
value = value_iteration(grid_world, initial_value)
policy = greedy_policy(grid_world, value)
print_value(value)
print_policy(policy)
print('----------------------------------------------------------------\n')

# Testing policy iteration
print('Policy iteration:')
policy = random_policy(grid_world)
policy[goal_state[0], goal_state[1], STOP] = 1.0
policy[goal_state[0], goal_state[1], UP:NUM_ACTIONS] = np.zeros(NUM_ACTIONS - 1)
value, policy = policy_iteration(grid_world, initial_value, policy)
print_value(value)
print_policy(policy)
print('----------------------------------------------------------------\n')
