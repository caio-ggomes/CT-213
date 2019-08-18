import numpy as np
import matplotlib.pyplot as plt
from utils import sum_gt_zero, xor
from keras import models, layers, losses, optimizers, activations, metrics, regularizers

# lambda_l2 = 0.000  # lambda parameter of the L2 regularization
lambda_l2 = 0.002  # lambda parameter of the L2 regularization
num_cases = 200  # number of auto-generated cases
num_epochs = 5000  # number of epochs for training
classification_function = sum_gt_zero  # selects sum_gt_zero as the classification function
# classification_function = xor  # selects xor as the classification function
# Figure format used for saving figures
fig_format = 'png'
# fig_format = 'svg'
# fig_format = 'eps'

if classification_function == sum_gt_zero:
    function_name = 'sgz'
else:
    function_name = 'xor'

# Setting the random seed of numpy's random library for reproducibility reasons
np.random.seed(0)

# Creating the dataset
inputs = np.zeros((num_cases, 2))
expected_outputs = np.zeros(num_cases)
for i in range(num_cases):
    inputs[i, :] = 10.0 * (-1.0 + 2.0 * np.random.rand(2))
    expected_outputs[i] = classification_function(inputs[i, :])
    inputs[i, :] += 1.0 * (-1.0 + 2.0 * np.random.randn(2))  # adding noise to corrupt the dataset


# Separating the dataset into positive and negative samples
positives_indices = np.where(expected_outputs >= 0.5)
negatives_indices = np.where(expected_outputs < 0.5)
positives_array = inputs[positives_indices, :]
negatives_array = inputs[negatives_indices, :]
positives_array = np.matrix(positives_array)
negatives_array = np.matrix(negatives_array)

# Creates the neural network model in Keras
model = models.Sequential()

# Adds the first layer
# The first argument refers to the number of neurons in this layer
# 'activation' configures the activation function
# input_shape represents the size of the input
# kernel_regularizer configures regularization for this layer
model.add(layers.Dense(50, activation=activations.sigmoid, input_shape=(2,),
                       kernel_regularizer=regularizers.l2(lambda_l2)))
model.add(layers.Dense(1, activation=activations.sigmoid, kernel_regularizer=regularizers.l2(lambda_l2)))

model.compile(optimizer=optimizers.Adam(), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])

history = model.fit(inputs, expected_outputs, batch_size=(num_cases // 4), epochs=num_epochs)

# Plotting cost function convergence
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.grid()
plt.savefig('convergence_' + function_name + '_l' + str(lambda_l2) + '.' + fig_format, format=fig_format)

# Plotting positive and negative samples
plt.figure()
plt.plot(positives_array[:, 0], positives_array[:, 1], '+r')
plt.plot(negatives_array[:, 0], negatives_array[:, 1], 'x')
plt.xlim([-10.0, 10.0])
plt.ylim([-10.0, 10.0])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Dataset')
plt.savefig('dataset_' + function_name + '_l' + str(lambda_l2) + '.' + fig_format, format=fig_format)


# Plotting the decision regions of the neural network
plt.figure()
x = np.arange(-10.0, 10.05, 0.05)
y = np.arange(-10.0, 10.05, 0.05)
X, Y = np.meshgrid(x, y)
x = np.reshape(X, (np.size(X, 0) * np.size(X, 1), 1))
y = np.reshape(Y, (np.size(Y, 0) * np.size(Y, 1), 1))
input = np.concatenate((x, y), axis=1)
z = model.predict(input)
Z = z.reshape((np.size(X, 0), np.size(Y, 0)))
plt.contourf(X, Y, Z)
plt.xlim([-10.0, 10.0])
plt.ylim([-10.0, 10.0])
plt.plot(positives_array[:, 0], positives_array[:, 1], '+', color='tab:orange')
plt.plot(negatives_array[:, 0], negatives_array[:, 1], 'x')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Neural Network Classification')
plt.savefig('nn_classification_' + function_name + '_l' + str(lambda_l2) + '.' + fig_format, format=fig_format)
plt.show()
