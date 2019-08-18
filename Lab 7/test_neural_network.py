import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from utils import sum_gt_zero, xor


num_cases = 200  # number of auto-generated cases
num_epochs = 1000  # number of epochs for training
# classification_function = sum_gt_zero  # selects sum_gt_zero as the classification function
classification_function = xor  # selects xor as the classification function
# Figure format used for saving figures
fig_format = 'png'
# fig_format = 'svg'
# fig_format = 'eps'

# Setting the random seed of numpy's random library for reproducibility reasons
np.random.seed(0)

# Creating the dataset
inputs = [None] * num_cases
expected_outputs = [None] * num_cases
for i in range(num_cases):
    inputs[i] = 5.0 * np.matrix(-1.0 + 2.0 * np.random.rand(2, 1))
    expected_outputs[i] = np.matrix(classification_function(inputs[i]))

# Separating the dataset into positive and negative samples
positives = []
negatives = []
for i in range(num_cases):
    if expected_outputs[i] >= 0.5:
        positives.append(inputs[i])
    else:
        negatives.append(inputs[i])
positives_array = np.zeros((2, len(positives)))
negatives_array = np.zeros((2, len(negatives)))
for i in range(len(positives)):
    positives_array[0, i] = positives[i][0, 0]
    positives_array[1, i] = positives[i][1, 0]
for i in range(len(negatives)):
    negatives_array[0, i] = negatives[i][0, 0]
    negatives_array[1, i] = negatives[i][1, 0]

# Creating and training the neural network
neural_network = NeuralNetwork(2, 10, 1, 6.0)
costs = np.zeros(num_epochs)
for i in range(num_epochs):
    neural_network.back_propagation(inputs, expected_outputs)
    costs[i] = neural_network.compute_cost(inputs, expected_outputs)
    print('epoch: %d; cost: %f' % (i + 1, costs[i]))

# Plotting cost function convergence
plt.plot(costs)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.grid()
plt.savefig('cost_function_convergence.' + fig_format, format=fig_format)

# Plotting positive and negative samples
plt.figure()
plt.plot(positives_array[0, :], positives_array[1, :], '+r')
plt.plot(negatives_array[0, :], negatives_array[1, :], 'x')
plt.xlim([-5.0, 5.0])
plt.ylim([-5.0, 5.0])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Dataset')
plt.savefig('dataset.' + fig_format, format=fig_format)


# Plotting the decision regions of the neural network
plt.figure()
x = np.arange(-5.0, 5.05, 0.05)
y = np.arange(-5.0, 5.05, 0.05)
z = np.zeros((len(x), len(y)))
for j in range(len(x)):
    for k in range(len(y)):
        _, a = neural_network.forward_propagation(np.matrix([[x[j]], [y[k]]]))
        z[j, k] = np.asscalar(a[-1])
plt.contourf(x, y, z)
plt.xlim([-5.0, 5.0])
plt.ylim([-5.0, 5.0])
plt.plot(positives_array[0, :], positives_array[1, :], '+', color='tab:orange')
plt.plot(negatives_array[0, :], negatives_array[1, :], 'x')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Neural Network Classification')
plt.savefig('neural_net_classification.' + fig_format, format=fig_format)
plt.show()
