from simple_evolution_strategy import SimpleEvolutionStrategy
import numpy as np
import matplotlib.pyplot as plt
from benchmark import translated_sphere, ackley, schaffer2d, rastrigin
import cma

# num_trials is the number of trials used in Monte Carlo
# num_iterations is the number of generations the algorithms are executed during a Monte Carlo trial
# num_trials = 500  # recommended for schaffer2d
num_trials = 200  # recommended for all other functions
# num_iterations = 200  # recommended for schaffer2d
num_iterations = 100  # recommended for all other functions
function = rastrigin  # translated_sphere, ackley, schaffer2d, rastrigin
fig_format = 'png'  # 'svg' (Word), 'eps' (Latex), 'png' (best compatibility/worst quality)


class Params:
    """
    An auxiliary class for storing parameters.
    """
    pass


def benchmark_algorithm(num_trials, num_iterations, algorithm, function, hyperparams):
    """
    Benchmarks an evolution strategy algorithm using Monte Carlo (MC) simulations.

    :param num_trials: number of Monte Carlo runs.
    :type num_trials: int.
    :param num_iterations: number of iterations.
    :type num_iterations: int.
    :param algorithm: evolution strategy algorithm (ses or cmaes).
    :type algorithm: str.
    :param function: function used for benchmarking.
    :type function: function.
    :param hyperparams: hyperparams of the algorithm.
    :type hyperparams: Params.
    :return mean_fitness: array containing the mean fitness of samples at each iteration (averaged over all MC trials).
    :rtype mean_fitness: numpy array of floats.
    :rtype best_fitness: array containing the best fitness of samples at each iteration (averaged over all MC trials).
    :rtype best_fitness: numpy array of floats.
    """
    if algorithm == 'ses':
        benchmark_name = '(%d,%d)-SES' % (hyperparams.mu, hyperparams.population_size)
    else:
        benchmark_name = 'CMA-ES'
    mean_fitness = [0.0] * num_iterations
    best_fitness = [0.0] * num_iterations
    for k in range(num_trials):
        # Printing the benchmarking status for the user
        print('%s Trial: %d/%d' % (benchmark_name, k + 1, num_trials))
        # Sampling the initial guess using an uniform distribution
        m0 = np.random.uniform(np.random.uniform(hyperparams.lower, hyperparams.upper))
        if algorithm == 'ses':
            C0 = hyperparams.C0
            es = SimpleEvolutionStrategy(m0, C0, hyperparams.mu, hyperparams.population_size)
        else:
            es = cma.CMAEvolutionStrategy(m0, hyperparams.sigma0)
        for i in range(num_iterations):
            samples = es.ask()
            if algorithm == 'ses':
                fitnesses = np.zeros(np.size(samples, 0))
                for j in range(np.size(samples, 0)):
                    fitnesses[j] = function(samples[j, :])
                es.tell(fitnesses)
            else:
                fitnesses = [function(sample) for sample in samples]
                es.tell(samples, fitnesses)
            mean_fitness[i] += np.mean(fitnesses)
            best_fitness[i] += np.min(fitnesses)
    for i in range(num_iterations):
        mean_fitness[i] /= num_iterations
        best_fitness[i] /= num_iterations
    return mean_fitness, best_fitness


hyperparams = Params()
# lower and upper are used for sampling the initial guess
hyperparams.lower = np.array([-3.0, -3.0])  # SES and CMA-ES
hyperparams.upper = np.array([3.0, 3.0])  # SES and CMA-ES
hyperparams.C0 = np.identity(2)  # SES only
hyperparams.sigma0 = 1.0  # CMA-ES only

# The default CMA-ES strategy (which is used here) uses mu = 3 and population_size = 6
hyperparams.mu = 3  # SES only
hyperparams.population_size = 6  # SES only
# Benchmarking (3,6)-SES
mean_ses6, best_ses6 = benchmark_algorithm(num_trials, num_iterations, 'ses', function, hyperparams)
hyperparams.mu = 6  # SES only
hyperparams.population_size = 12  # SES only
# Benchmarking (6,12)-SES
mean_ses12, best_ses12 = benchmark_algorithm(num_trials, num_iterations, 'ses', function, hyperparams)
hyperparams.mu = 12  # SES only
hyperparams.population_size = 24  # SES only
# Benchmarking (12,24)-SES
mean_ses24, best_ses24 = benchmark_algorithm(num_trials, num_iterations, 'ses', function, hyperparams)
# Benchmarking (3_w,6)-CMA-ES
mean_cmaes, best_cmaes = benchmark_algorithm(num_trials, num_iterations, 'cmaes', function, hyperparams)
plt.figure()
plt.plot(mean_ses6)
plt.plot(mean_ses12)
plt.plot(mean_ses24)
plt.plot(mean_cmaes)
plt.legend(['(3,6)-SES', '(6,12)-SES', '(12,24)-SES', 'CMA-ES'])
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.title('Mean Fitness - %s' % function.__name__)
plt.savefig('mean_fitness.%s' % fig_format, fig_format=fig_format)
plt.figure()
plt.plot(best_ses6)
plt.plot(best_ses12)
plt.plot(best_ses24)
plt.plot(best_cmaes)
plt.legend(['(3,6)-SES', '(6,12)-SES', '(12,24)-SES', 'CMA-ES'])
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.title('Best Fitness - %s' % function.__name__)
plt.savefig('best_fitness.%s' % fig_format, fig_format=fig_format)
plt.show()