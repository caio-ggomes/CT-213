import numpy as np
import random
from math import pi, cos, sin
from least_squares import least_squares
from gradient_descent import gradient_descent
from hill_climbing import hill_climbing
from simulated_annealing import simulated_annealing
import matplotlib.pyplot as plt


def cost_function(theta):
    """
    Samples the linear regression cost function.

    :param theta: parameter point.
    :type theta: numpy.array.
    :return: cost value at theta.
    :rtype: float.
    """
    return sum((theta[0] + theta[1] * t - v) ** 2) / (2.0 * m)


def gradient_function(theta):
    """
    Samples the gradient of the linear regression cost function.

    :param theta: parameter point.
    :type theta: numpy.array.
    :return: gradient at theta.
    :rtype: float.
    """
    return np.array([(1 / m) * sum(theta[0] + theta[1] * t - v),
                    (1 / m) * sum((theta[0] + theta[1] * t - v) * t)])


def fit_least_squares():
    """
    Uses the Least Squares Method to fit the ball parameters.

    :return: array containing the initial speed and the acceleration factor due to rolling friction.
    :rtype: numpy.array.
    """
    return least_squares([lambda x: 1.0, lambda x: x], t, v)


def fit_gradient_descent():
    """
    Uses Gradient Descent (GD) to fit the ball parameters.

    :return theta: array containing the initial speed and the acceleration factor due to rolling friction.
    :rtype theta: numpy.array.
    :return history: history of points visited by the algorithm.
    :rtype history: list of numpy.array.
    """
    theta, history = gradient_descent(cost_function, gradient_function, np.array([0.0, 0.0]), 0.1, 1.0e-10, 1000)
    return theta, history


def fit_hill_climbing():
    """
    Uses Hill Climbing (HC) to fit the ball parameters.

    :return theta: array containing the initial speed and the acceleration factor due to rolling friction.
    :rtype theta: numpy.array.
    :return history: history of points visited by the algorithm.
    :rtype history: list of numpy.array.
    """
    # Hyperparameters used for computing the neighbors
    delta = 2.0e-3
    num_neighbors = 8

    def neighbors(theta):
        """
        Returns 8-connected neighbors of point theta.
        The neighbors are sampled around a circle of radius "delta".
        Equally spaced (in terms of angle) "num_neighbors" neighbors are sampled.

        :param theta: current point.
        :type theta: numpy.array.
        :return: neighbors of theta.
        :rtype: list of numpy.array.
        """
        neighbors_list = []
        # Todo: Implement
        i = 0
        while i < num_neighbors:
            neighbors_list.append(np.array([theta[0] + delta * cos(2 * pi * i / num_neighbors), theta[1] + delta * sin(2 * pi * i / num_neighbors)]))
            i = i + 1
        return neighbors_list

    theta, history = hill_climbing(cost_function, neighbors, np.array([0.0, 0.0]), 1.0e-10, 1000)
    return theta, history


def fit_simulated_annealing():
    """
    Uses Simulated Annealing (SA) to fit the ball parameters.

    :return theta: array containing the initial speed and the acceleration factor due to rolling friction.
    :rtype theta: numpy.array.
    :return history: history of points visited by the algorithm.
    :rtype history: list of numpy.array.
    """
    # Hyperparameter used for computing the random neighbor
    delta = 2.0e-3
    # Hyperparameters used for computing the temperature scheduling
    temperature0 = 1.0
    beta = 1.0

    def random_neighbor(theta):
        """
        Returns a random neighbor of theta.
        The random neighbor is sampled around a circle of radius <delta>.
        The probability distribution of the angle is uniform(-pi, pi).

        :param theta: current point.
        :type theta: numpy.array.
        :return: random neighbor.
        :rtype: numpy.array.
        """
        # Todo: Implement
        angle = random.uniform(-pi, pi)
        return np.array([theta[0] + delta*cos(angle), theta[1] + delta*sin(angle)])

    def schedule(i):
        """
        Defines the temperature schedule of the simulated annealing.

        :param i: current iteration.
        :type i: int.
        :return: current temperature.
        :rtype: float.
        """
        # Todo: Implement
        return temperature0 / (1+beta*i**2)

    theta, history = simulated_annealing(cost_function, random_neighbor, schedule, np.array([0.0, 0.0]), 1.0e-10, 5000)
    return theta, history


def plot_optimization(history):
    """
    Plots the optimization history.

    :param history: points visited by the optimization algorithm.
    :type history: list of numpy.array.
    """
    t0 = np.arange(-0.5, 0.5, 0.01)
    t1 = np.arange(-0.5, 0.5, 0.01)
    z = np.zeros((len(t0), len(t1)))
    for i in range(len(t0)):
        for j in range(len(t1)):
            z[i, j] = cost_function(np.array([t0[i], t1[j]]))
    plt.contourf(t0, t1, z.transpose())
    hx = []
    hy = []
    for h in history:
        hx.append(h[0])
        hy.append(h[1])
    handle, = plt.plot(hx, hy, '.-', markersize=5)
    plt.xlabel('v_0 (m/s)')
    plt.ylabel('f (m/s^2)')
    plt.plot(hx[0], hy[0], '*y')
    plt.plot(hx[-1], hy[-1], 'xr')
    return handle


fig_format = 'png'
# fig_format = 'svg'
# fig_format = 'eps'
# Recommended figure formats: .eps for Latex/Linux, .svg for MS Office, and .png for easy visualization in Windows.
# The quality of .eps and .svg is far superior since these are vector graphics formats.

# Setting random seed for reproducibility
random.seed(100)
# Loading and pre-processing data
data = np.genfromtxt('data.txt')
t = data[:, 0]
x = data[:, 1]
y = data[:, 2]
t -= t[0]
m = len(t)
vx = np.zeros(m)
vy = np.zeros(m)
vx[0] = (x[1] - x[0]) / (t[1] - t[0])
vy[0] = (y[1] - y[0]) / (t[1] - t[0])
for k in range(1, m - 1):
    vx[k] = (x[k + 1] - x[k - 1]) / (t[k + 1] - t[k - 1])
    vy[k] = (y[k + 1] - y[k - 1]) / (t[k + 1] - t[k - 1])
vx[-1] = (x[-1] - x[-2]) / (t[-1] - t[-2])
vy[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
v = np.sqrt(vx ** 2 + vy ** 2)

# Solving the problem using Least Squares in order to obtain ground truth
theta_ls = fit_least_squares()
print('Least Squares solution: ', theta_ls)
# Solving the problem using each algorithm and plotting the optimization history of each algorithm
theta_gd, history_gd = fit_gradient_descent()
print('Gradient Descent solution: ', theta_gd)
plt.figure()
plot_optimization(history_gd)
plt.title('Gradient Descent')
plt.savefig('gradient_descent.%s' % fig_format, format=fig_format)
theta_hc, history_hc = fit_hill_climbing()
print('Hill Climbing solution: ', theta_hc)
plt.figure()
plot_optimization(history_hc)
plt.title('Hill Climbing')
plt.savefig('hill_climbing.%s' % fig_format, format=fig_format)
theta_sa, history_sa = fit_simulated_annealing()
print('Simulated Annealing solution: ', theta_sa)
plt.figure()
plot_optimization(history_sa)
plt.title('Simulated Annealing')
plt.savefig('simulated_annealing.%s' % fig_format, format=fig_format)
# Tracing the optimization histories in a single plot for comparison
plt.figure()
handle_gd = plot_optimization(history_gd)
handle_hc = plot_optimization(history_hc)
handle_sa = plot_optimization(history_sa)
plt.legend([handle_gd, handle_hc, handle_sa], ['Gradient Descent', 'Hill Climbing', 'Simulated Annealing'])
plt.title('Optimization Comparison')
plt.savefig('optimization_comparison.%s' % fig_format, format=fig_format)

# Plotting the curve fit
plt.figure()
plt.plot(t, v, '*k')
v_ls = theta_ls[0] + theta_ls[1] * t
v_gd = theta_gd[0] + theta_gd[1] * t
v_hc = theta_hc[0] + theta_hc[1] * t
v_sa = theta_sa[0] + theta_sa[1] * t
plt.plot(t, v_ls, 'tab:red')
plt.plot(t, v_gd, 'tab:blue')
plt.plot(t, v_hc, 'tab:orange')
plt.plot(t, v_sa, 'tab:green')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.legend(['Data Points', 'Least Squares', 'Gradient Descent', 'Hill Climbing', 'Simulated Annealing'])
plt.title('Curve Fit')
plt.savefig('fit_comparison.%s' % fig_format, format=fig_format)
plt.show()
