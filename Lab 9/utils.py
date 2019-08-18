import gzip
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt


def read_mnist(images_path, labels_path):
    """
    Loads the MNIST dataset.

    :param images_path: the path of the file containing the images.
    :type images_path: str.
    :param labels_path: the path of the file containing the labels.
    :type labels_path: str.
    :return features: images.
    :rtype features: numpy.ndarray.
    :return labels: labels.
    :rtype labels: numpy.ndarray.
    """
    with gzip.open(labels_path, 'rb') as labels_file:
        labels = np.frombuffer(labels_file.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as images_file:
        length = len(labels)
        # Load flat 28x28 px images (784 px), and convert them to 28x28 px
        features = np.frombuffer(images_file.read(), dtype=np.uint8, offset=16) \
            .reshape(length, 784) \
            .reshape(length, 28, 28, 1)

    # Pad images with 0s
    features = np.pad(features, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

    return features, labels


def display_image(image, title):
    """
    Displays a NMIST image.

    :param image: the image to be displayed.
    :type image: numpy.ndarray.
    :param title: the figure's title.
    :type title: str.
    """
    image = image.squeeze()
    plt.figure()
    plt.title(title)
    plt.imshow(image, cmap=plt.cm.gray_r)


def save_model_to_json(model, model_name):
    """
    Saves a Keras' model in JSON format.

    :param model: Keras' model to be saved.
    :param model_name: the name used for the model's files.
    :type model_name: str.
    """
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_name + '.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_name + '.h5')


def load_model_from_json(model_name):
    """
    Loads a Keras' model from JSON format.

    :param model_name: the used in the model's files.
    :type model_name: str.
    :return: loaded Keras' model.
    """
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name + '.h5')
    return loaded_model

