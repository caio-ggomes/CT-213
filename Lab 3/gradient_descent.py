def gradient_descent(cost_function, gradient_function, theta0, alpha, epsilon, max_iterations):
    """
    Executes the Gradient Descent (GD) algorithm to minimize (optimize) a cost function.

    :param cost_function: function to be minimized.
    :type cost_function: function.
    :param gradient_function: gradient of the cost function.
    :type gradient_function: function.
    :param theta0: initial guess.
    :type theta0: numpy.array.
    :param alpha: learning rate.
    :type alpha: float.
    :param epsilon: used to stop the optimization if the current cost is less than epsilon.
    :type epsilon: float.
    :param max_iterations: maximum number of iterations.
    :type max_iterations: int.
    :return theta: local minimum.
    :rtype theta: numpy.array.
    :return history: history of points visited by the algorithm.
    :rtype history: list of numpy.array.
    """
    theta = theta0
    history = [theta0]
    # Todo: Implement Gradient Descent
    i = 0
    while i < max_iterations and cost_function(theta) > epsilon:
        theta = theta - alpha*gradient_function(theta)
        history.append(theta)
        i = i+1
    return theta, history
