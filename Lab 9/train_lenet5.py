import os
from time import time

from sklearn.model_selection import train_test_split
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard
from lenet5 import make_lenet5
from utils import read_mnist, save_model_to_json

# Comment this line to enable training using your GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

EPOCHS = 10  # number of epochs used for training
BATCH_SIZE = 128  # size of mini-batch

# Load the training dataset
train_features, train_labels = read_mnist('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')

# Splitting the dataset into training and (cross-)validation datasets
train_features, validation_features, train_labels, validation_labels = \
    train_test_split(train_features, train_labels, test_size=0.2, random_state=0)

print('# of training images:', train_features.shape[0])
print('# of cross-validation images:', validation_features.shape[0])

model = make_lenet5()
# Shows a summary of the CNN to verify if it is correctly implemented
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# Preparing the datasets for training
X_train, y_train = train_features, to_categorical(train_labels)
X_validation, y_validation = validation_features, to_categorical(validation_labels)


train_generator = ImageDataGenerator().flow(X_train, y_train, batch_size=BATCH_SIZE)
validation_generator = ImageDataGenerator().flow(X_validation, y_validation, batch_size=BATCH_SIZE)

steps_per_epoch = X_train.shape[0] // BATCH_SIZE
validation_steps = X_validation.shape[0] // BATCH_SIZE

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
                    validation_data=validation_generator, validation_steps=validation_steps,
                    shuffle=True, callbacks=[tensorboard])

# Save the trained model to files
save_model_to_json(model, 'lenet5')
