import numpy as np
from math import inf, fabs
from utils import *


def random_policy(grid_world):
    """
    Creates a random policy for a grid world.

    :param grid_world: the grid world.
    :type grid_world: GridWorld.
    :return: random policy.
    :rtype: tridimensional NumPy array.
    """
    dimensions = grid_world.dimensions
    policy = (1.0 / NUM_ACTIONS) * np.ones((dimensions[0], dimensions[1], NUM_ACTIONS))
    return policy


def greedy_policy(grid_world, value, epsilon=1.0e-3):
    """
    Computes a greedy policy considering a value function for a grid world. If there are more than
    one optimal action for a given state, then the optimal action is chosen at random.


    :param grid_world: the grid world.
    :type grid_world: GridWorld.
    :param value: the value function.
    :type value: bidimensional NumPy array.
    :param epsilon: tolerance used to consider that more than one action is optimal.
    :type epsilon: float.
    :return: greedy policy.
    :rtype: tridimensional NumPy array.
    """
    dimensions = grid_world.dimensions
    policy = np.zeros((dimensions[0], dimensions[1], NUM_ACTIONS))
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            current_state = (i, j)
            if not grid_world.is_cell_valid(current_state):
                # Assuming random action if the cell is an obstacle
                policy[i, j] = (1.0 / NUM_ACTIONS) * np.ones(NUM_ACTIONS)
                continue
            max_value = -inf
            action_value = np.zeros(NUM_ACTIONS)  # Creating a temporary q(s, a)
            for action in range(NUM_ACTIONS):
                r = grid_world.reward(current_state, action)
                action_value[action] = r
                for next_state in grid_world.get_valid_sucessors((i, j), action):
                    transition_prob = grid_world.transition_probability(current_state, action, next_state)
                    action_value[action] += grid_world.gamma * transition_prob * value[next_state[0], next_state[1]]
                if action_value[action] > max_value:
                    max_value = action_value[action]
            # This post-processing is necessary since we may have more than one optimal action
            num_actions = 0
            for action in range(NUM_ACTIONS):
                if fabs(max_value - action_value[action]) < epsilon:
                    policy[i, j, action] = 1.0
                    num_actions += 1
            for action in range(NUM_ACTIONS):
                policy[i, j, action] /= num_actions
    return policy


def policy_evaluation(grid_world, initial_value, policy, num_iterations=10000, epsilon=1.0e-5):
    """
    Executes policy evaluation for a policy executed on a grid world.

    :param grid_world: the grid world.
    :type grid_world: GridWorld.
    :param initial_value: initial value function used to bootstrap the algorithm.
    :type initial_value: bidimensional NumPy array.
    :param policy: policy to be evaluated.
    :type policy: tridimensional NumPy array.
    :param num_iterations: maximum number of iterations used in policy evaluation.
    :type num_iterations: int.
    :param epsilon: tolerance used in stopping criterion.
    :type epsilon: float.
    :return: value function of the given policy.
    :rtype: bidimensional NumPy array.
    """
    dimensions = grid_world.dimensions
    value = np.copy(initial_value)
    # Todo: implement policy evaluation.
    old_value = np.zeros((dimensions[0], dimensions[1]))
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            old_value[i, j] = value[i, j] - 2*epsilon
    iteration = 0
    policy_reward = np.zeros((dimensions[0], dimensions[1]))
    policy_probability = np.zeros((dimensions[0], dimensions[1], dimensions[0], dimensions[1]))
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            current_state = (i, j)
            policy_reward[current_state] = 0
            for action in range(NUM_ACTIONS):
                policy_reward[current_state] += policy[current_state][action]*grid_world.reward(current_state, action)
                for other_i in range(dimensions[0]):
                    for other_j in range(dimensions[1]):
                        other_state = (other_i, other_j)
                        policy_probability[current_state][other_state] += policy[current_state][action]*grid_world.transition_probability(current_state, action, other_state)
    diff = value - old_value
    while max(abs(diff.max()), abs(diff.min())) > epsilon and iteration < num_iterations:
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                old_value[i, j] = value[i, j]
        matrix = np.zeros((dimensions[0], dimensions[1]))
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                current_state = (i, j)
                for other_i in range(dimensions[0]):
                    for other_j in range(dimensions[1]):
                        other_state = (other_i, other_j)
                        matrix[current_state] += policy_probability[current_state][other_state]*old_value[other_state]
        value = policy_reward + grid_world.gamma*matrix
        iteration += 1
        diff = value - old_value
    return value


def value_iteration(grid_world, initial_value, num_iterations=10000, epsilon=1.0e-5):
    """
    Executes value iteration for a grid world.

    :param grid_world: the grid world.
    :type grid_world: GridWorld.
    :param initial_value: initial value function used to bootstrap the algorithm.
    :type initial_value: bidimensional NumPy array.
    :param num_iterations: maximum number of iterations used in policy evaluation.
    :type num_iterations: int.
    :param epsilon: tolerance used in stopping criterion.
    :type epsilon: float.
    :return value: optimal value function.
    :rtype value: bidimensional NumPy array.
    """
    dimensions = grid_world.dimensions
    value = np.copy(initial_value)
    # Todo: implement value iteration.
    iteration = 0
    old_value = np.zeros((dimensions[0], dimensions[1]))
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            old_value[i, j] = value[i, j] - 2*epsilon
    diff = value - old_value
    while max(abs(diff.max()), abs(diff.min())) > epsilon and iteration < num_iterations:
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                old_value[i, j] = value[i, j]
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                current_state = (i, j)
                maximum = -inf
                for action in range(NUM_ACTIONS):
                    potencial_value = 0
                    potencial_value += grid_world.reward(current_state, action)
                    for other_i in range(dimensions[0]):
                        for other_j in range(dimensions[1]):
                            other_state = (other_i, other_j)
                            potencial_value += grid_world.gamma * grid_world.transition_probability(current_state, action, other_state) * old_value[other_state]
                    if potencial_value > maximum:
                        maximum = potencial_value
                value[current_state] = maximum
        iteration += 1
        diff = value - old_value
    return value


def policy_iteration(grid_world, initial_value, initial_policy, evaluations_per_policy=3, num_iterations=10000,
                     epsilon=1.0e-5):
    """
    Executes policy iteration for a grid world.

    :param grid_world: the grid world.
    :type grid_world: GridWorld.
    :param initial_value: initial value function used to bootstrap the algorithm.
    :type initial_value: bidimensional NumPy array.
    :param initial_policy: initial policy used to bootstrap the algorithm.
    :type initial_policy: tridimensional NumPy array.
    :param evaluations_per_policy: number of policy evaluations per policy iteration.
    :type evaluations_per_policy: int.
    :param num_iterations: maximum number of iterations used in policy evaluation.
    :type num_iterations: int.
    :param epsilon: tolerance used in stopping criterion.
    :type epsilon: float.
    :return value: value function of the optimal policy.
    :rtype value: bidimensional NumPy array.
    :return policy: optimal policy.
    :rtype policy: tridimensional NumPy array.
    """
    value = np.copy(initial_value)
    policy = np.copy(initial_policy)
    # Todo: implement policy iteration.
    dimensions = grid_world.dimensions
    old_value = np.zeros((dimensions[0], dimensions[1]))
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            old_value[i, j] = value[i, j] - 2*epsilon
    iteration = 0
    diff = value - old_value
    while max(abs(diff.max()), abs(diff.min())) > epsilon and iteration < num_iterations:
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                old_value[i, j] = value[i, j]
        value = policy_evaluation(grid_world, value, policy, evaluations_per_policy)
        policy = greedy_policy(grid_world, value)
        iteration += 1
        diff = value - old_value
    return value, policy

