from utils import read_mnist, display_image
import random
import matplotlib.pyplot as plt


NUM_IMAGES = 10

# Loading training and test datasets
train_features, train_labels = read_mnist('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
test_features, test_labels = read_mnist('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')

print('# of training images:', train_features.shape[0])
print('# of test images:', test_features.shape[0])

print('Training dataset shape:', train_features.shape)
print('Test dataset shape:', test_features.shape)

# Showing some random images
for i in range(NUM_IMAGES):
    index = random.randint(0, train_features.shape[0])
    display_image(train_features[index], 'Example: %d. Label: %d' % (index, train_labels[index]))

plt.show()
