import numpy as np


def sigmoid(x):
    """
    Sigmoid function.

    :param x: input to the function.
    :type x: float or numpy matrix.
    :return: output of the sigmoid function evaluated at x.
    :rtype x: float or numpy matrix.
    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Sigmoid function derivative.

    :param x: input to the function.
    :type x: float or numpy matrix.
    :return: derivative of the sigmoid function evaluated at x.
    :rtype: float or numpy matrix.
    """
    return np.multiply(sigmoid(x), (1.0 - sigmoid(x)))


def signal(x):
    """
    Returns the signal of the input as 1.0 or -1.0.

    :param x: input to the function.
    :type x: float.
    :return: signal of x.
    :rtype x: float.
    """
    if x >= 0.0:
        return 1.0
    return -1.0


def sum_gt_zero(x):
    """
    Returns 1.0 if the sum of the coordinates of x is greater than 0.
    Otherwise, returns 0.0. This function only works for 2D inputs.

    :param x: input to the function.
    :type x: 2x1 numpy matrix.
    :return: 1.0 if the sum of the input coordinates is greater than 0, 0.0 otherwise.
    :rtype: float.
    """
    s = x[0] + x[1]
    if s > 0.0:
        return 1.0
    return 0.0


def xor(x):
    """
    Implements a XOR-like function using the signals of the input coordinates.
    Returns 1.0 if the signal of the two coordinates are the same.
    Otherwise, returns 0.0. This function only works for 2D inputs.

    :param x: input to the function.
    :type x: 2x1 numpy matrix.
    :return: 1.0 if the signal of the two coordinates are the same, 0.0 otherwise.
    :type: float.
    """
    if signal(x[0]) == signal(x[1]):
        return 1.0
    return 0.0

