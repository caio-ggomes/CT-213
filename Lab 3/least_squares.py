import numpy as np


def least_squares(phi, x, y):
    """
    Executes the Least Squares Method in order to fit a model to a collection of data points (x, y).

    :param phi: functions of the model.
    :type phi: list of functions.
    :param x: independent variable.
    :type x: numpy.array of float.
    :param y: dependent variable.
    :type y: numpy.array of float.
    :return: least square estimate of the model parameters.
    :rtype: numpy.array of float.
    """
    m = len(x)
    n = len(phi) - 1
    a = np.zeros((n + 1, n + 1))
    b = np.zeros((n + 1))
    # Making matrix A
    for k in range(n + 1):
        for j in range(n + 1):
            for i in range(m):
                a[k, j] += phi[k](x[i]) * phi[j](x[i])

    # Making vector b
    for j in range(n + 1):
        for i in range(m):
            b[j] += phi[j](x[i]) * y[i]

    # Solving system A * x = b
    theta = np.linalg.solve(a, b)
    return theta
