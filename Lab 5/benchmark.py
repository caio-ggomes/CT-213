import numpy as np
from math import sin, cos, sqrt, exp, pi


def translated_sphere(x):
    """
    Implements a n-dimensional translated sphere function for benchmarking optimization algorithms.
    The center of the sphere is at (1,2,3,...,n).

    :param x: point to be evaluated.
    :type x: numpy array of floats.
    :return: function value at x.
    :rtype: float.
    """
    num_dimensions = np.size(x)
    sum = 0.0
    for i in range(num_dimensions):
        sum += (x[i] - i - 1) ** 2
    return sum


def ackley(x):
    """
    Implements the Ackley function for benchmarking optimization algorithms.

    :param x: point to be evaluated.
    :type x: numpy array of floats.
    :return: function value at x.
    :rtype: float.
    """
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))) - \
           exp(0.5 * (cos(2 * pi * x[0]) + cos(2.0 * pi * x[1]))) + exp(1) + 20.0


def schaffer2d(x):
    """
    Implements the Schaffer2D function for benchmarking optimization algorithms.

    :param x: point to be evaluated.
    :type x: numpy array of floats.
    :return: function value at x.
    :rtype: float.
    """
    return 0.5 + (sin(x[0] ** 2 - x[1] ** 2) ** 2 - 0.5) / (1.0 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2


def rastrigin(x):
    """
    Implements the Rastrigin function for benchmarking optimization algorithms.

    :param x: point to be evaluated.
    :type x: numpy array of floats.
    :return: function value at x.
    :rtype: float.
    """
    n = np.size(x)
    a = 10.0
    sum = a * n
    for i in range(n):
        sum += x[i] ** 2 - a * cos(2 * pi * x[i])
    return sum
