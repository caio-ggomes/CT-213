from keras import layers, activations
from keras.models import Sequential


def make_lenet5():
    model = Sequential()

    # Todo: implement LeNet-5 model
    model.add(layers.Conv2D(filters=6, input_shape=(32, 32, 1), kernel_size=(5, 5), strides=1, activation=activations.tanh))
    model.add(layers.AveragePooling2D(input_shape=(28, 28, 1), pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(filters=16, input_shape=(14, 14, 1), kernel_size=(5, 5), strides=1, activation=activations.tanh))
    model.add(layers.AveragePooling2D(input_shape=(10, 10, 1), pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(filters=120, input_shape=(5, 5, 1), kernel_size=(5, 5), strides=1, activation=activations.tanh))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=84, activation=activations.tanh))
    model.add(layers.Dense(units=10, activation=activations.softmax))
    ######################
    return model
