import numpy as np


def sigmoid(x):
    """
    Computes the sigmoid activation function.

    :param x: input value.
    :type x: float.
    :return: output value.
    :rtype: float.
    """
    return 1.0 / (1.0 + np.exp(-x))

