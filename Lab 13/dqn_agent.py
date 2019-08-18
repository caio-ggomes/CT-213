import random
import numpy as np
from collections import deque
from keras import models, layers, optimizers, activations, losses


class DQNAgent:
    """
    Represents a Deep Q-Networks (DQN) agent.
    """
    def __init__(self, state_size, action_size, gamma=0.95, epsilon=0.5, epsilon_min=0.01, epsilon_decay=0.98, learning_rate=0.001, buffer_size=4098):
        """
        Creates a Deep Q-Networks (DQN) agent.

        :param state_size: number of dimensions of the feature vector of the state.
        :type state_size: int.
        :param action_size: number of actions.
        :type action_size: int.
        :param gamma: discount factor.
        :type gamma: float.
        :param epsilon: epsilon used in epsilon-greedy policy.
        :type epsilon: float.
        :param epsilon_min: minimum epsilon used in epsilon-greedy policy.
        :type epsilon_min: float.
        :param epsilon_decay: decay of epsilon per episode.
        :type epsilon_decay: float.
        :param learning_rate: learning rate of the action-value neural network.
        :type learning_rate: float.
        :param buffer_size: size of the experience replay buffer.
        :type buffer_size: int.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=buffer_size)  # giving a maximum length makes this buffer forget old memories
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self.make_model()

    def make_model(self):
        """
        Makes the action-value neural network model using Keras.

        :return: action-value neural network.
        :rtype: Keras' model.
        """
        # raise NotImplementedError('You need to implement the neural network model.')  # Remove this line
        # Todo: Uncomment the lines below
        model = models.Sequential()
        # Todo: implement Keras' model
        model.add(layers.Dense(input_dim=self.state_size, units=24, activation=activations.relu))
        model.add(layers.Dense(units=24, activation=activations.relu))
        model.add(layers.Dense(units=self.action_size, activation=activations.linear))
        model.compile(loss=losses.mse, optimizer=optimizers.Adam(lr=self.learning_rate))
        model.summary()
        return model

    def act(self, state):
        """
        Chooses an action using an epsilon-greedy policy.

        :param state: current state.
        :type state: NumPy array with dimension (1, 2).
        :return: chosen action.
        :rtype: int.
        """
        # Todo: implement epsilon-greey action selection.
        probability = np.random.uniform(0, 1)
        if probability < self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            return np.argmax(self.model.predict(state))
        # Todo: change this line

    def append_experience(self, state, action, reward, next_state, done):
        """
        Appends a new experience to the replay buffer (and forget an old one if the buffer is full).

        :param state: state.
        :type state: NumPy array with dimension (1, 2).
        :param action: action.
        :type action: int.
        :param reward: reward.
        :type reward: float.
        :param next_state: next state.
        :type next_state: NumPy array with dimension (1, 2).
        :param done: if the simulation is over after this experience.
        :type done: bool.
        """
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """
        Learns from memorized experience.

        :param batch_size: size of the minibatch taken from the replay buffer.
        :type batch_size: int.
        :return: loss computed during the neural network training.
        :rtype: float.
        """
        minibatch = random.sample(self.replay_buffer, batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if not done:
                target[0][action] = reward + self.gamma * np.max(self.model.predict(next_state)[0])
            else:
                target[0][action] = reward
            # Filtering out states and targets for training
            states.append(state[0])
            targets.append(target[0])
        history = self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        return loss

    def load(self, name):
        """
        Loads the neural network's weights from disk.

        :param name: model's name.
        :type name: str.
        """
        self.model.load_weights(name)

    def save(self, name):
        """
        Saves the neural network's weights to disk.

        :param name: model's name.
        :type name: str.
        """
        self.model.save_weights(name)

    def update_epsilon(self):
        """
        Updates the epsilon used for epsilon-greedy action selection.
        """
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min
