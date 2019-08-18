import numpy as np
import random
from math import inf


class Particle:
    """
    Represents a particle of the Particle Swarm Optimization algorithm.
    """
    def __init__(self, lower_bound, upper_bound):
        """
        Creates a particle of the Particle Swarm Optimization algorithm.

        :param lower_bound: lower bound of the particle position.
        :type lower_bound: numpy array.
        :param upper_bound: upper bound of the particle position.
        :type upper_bound: numpy array.
        """
        # Todo: implement
        self.x = np.zeros(np.size(lower_bound))
        self.v = np.zeros(np.size(lower_bound))
        self.value = -inf
        self.evaluated = False
        i = 0
        while i < np.size(lower_bound):
            self.x[i] = random.uniform(lower_bound[i], upper_bound[i])
            self.v[i] = random.uniform(-(upper_bound[i] - lower_bound[i]), upper_bound[i] - lower_bound[i])
            i = i + 1
        self.best_position = self.x
        self.best_value = -inf


class ParticleSwarmOptimization:
    """
    Represents the Particle Swarm Optimization algorithm.
    Hyperparameters:
        inertia_weight: inertia weight.
        cognitive_parameter: cognitive parameter.
        social_parameter: social parameter.

    :param hyperparams: hyperparameters used by Particle Swarm Optimization.
    :type hyperparams: Params.
    :param lower_bound: lower bound of particle position.
    :type lower_bound: numpy array.
    :param upper_bound: upper bound of particle position.
    :type upper_bound: numpy array.
    """
    def __init__(self, hyperparams, lower_bound, upper_bound):
        # Todo: implement
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.num_particles = hyperparams.num_particles
        self.w = hyperparams.inertia_weight
        self.phip = hyperparams.cognitive_parameter
        self.phig = hyperparams.social_parameter
        self.particles = np.array([])
        self.current_particle = None
        i = 0
        while i < self.num_particles:
            particle = Particle(lower_bound, upper_bound)
            self.particles = np.append(self.particles, particle)
            i = i + 1

    def get_best_position(self):
        """
        Obtains the best position so far found by the algorithm.

        :return: the best position.
        :rtype: numpy array.
        """
        # Todo: implement
        best_value_global = -inf
        position = None
        for particle in self.particles:
            if particle.best_value >= best_value_global:
                position = particle.best_position
                best_value_global = particle.best_value
        return position

    def get_best_value(self):
        """
        Obtains the value of the best position so far found by the algorithm.

        :return: value of the best position.
        :rtype: float.
        """
        # Todo: implement
        best_value_global = -inf
        for particle in self.particles:
            if particle.best_value >= best_value_global:
                best_value_global = particle.best_value
        return best_value_global  # Remove this line

    def get_position_to_evaluate(self):
        """
        Obtains a new position to evaluate.

        :return: position to evaluate.
        :rtype: numpy array.
        """
        # Todo: implement
        for particle in self.particles:
            if particle.evaluated is False:
                self.current_particle = particle
                return self.current_particle.x
        self.advance_generation()
        for particle in self.particles:
            if particle.evaluated is False:
                self.current_particle = particle
                return self.current_particle.x

    def advance_generation(self):
        """
        Advances the generation of particles.
        """
        # Todo: implement
        for particle in self.particles:
            if particle.value > particle.best_value:
                particle.best_position = particle.x
                particle.best_value = particle.value
            rp = random.uniform(0.0, 1.0)
            rg = random.uniform(0.0, 1.0)
            particle.v = self.w * particle.v + self.phip * rp * (particle.best_position - particle.x) + self.phig * rg * (self.get_best_position() - particle.x)
            particle.x = particle.x + particle.v
            particle.evaluated = False

    def notify_evaluation(self, value):
        """
        Notifies the algorithm that a particle position evaluation was completed.

        :param value: quality of the particle position.
        :type value: float.
        """
        # Todo: implement
        self.current_particle.value = value
        self.current_particle.evaluated = True
