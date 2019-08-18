import os
from yolo_detector import YoloDetector
import cv2
import matplotlib.pyplot as plt
import time

# Uncomment/comment this to disable/enable your GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

NUM_IMAGES = 10
DETECTION_THRESHOLD = 0.5  # probability threshold used to discard object detections
# Used for plotting purposes
RECT_THICKNESS = 5
CIRCLE_RADIUS = 5
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)


def load_image(image_name):
    """
    Loads an image.

    :param image_name: image name.
    :type image_name: str.
    :return: loaded image.
    :rtype: OpenCV's image.
    """
    exists = os.path.isfile(image_name + '.jpg')
    if exists:
        return cv2.imread(image_name + '.jpg')
    else:
        return cv2.imread(image_name + '.png')


def show_detection(image, detection, color, show_center=False):
    """
    Shows an object detection.

    :param image: image used for object detection.
    :type image: OpenCV's image.
    :param detection: detection as a 5-dimensional tuple: (probability, x, y, width, height).
    :type detection: 5-dimensional tuple.
    :param color: color used to show the detection (RGB value).
    :type color: 3-dimensional tuple.
    :param show_center: if the center of the detection should be shown in the image.
    :type show_center: bool.
    """
    x = detection[1]
    y = detection[2]
    width = detection[3]
    height = detection[4]
    top_left = (int(x - width / 2), int(y - height / 2))
    bottom_right = (int(x + width / 2), int(y + height / 2))
    cv2.rectangle(image, top_left, bottom_right, color, RECT_THICKNESS)
    if show_center:
        cv2.circle(image, (int(x), int(y)), CIRCLE_RADIUS, color, -1)


def show_detections(image_name, image, ball_detection, post1_detection, post2_detection):
    """
    Shows the image with all relevant detections.

    :param image_name: image name used for the plot's title.
    :type image_name: str.
    :param image: image used for object detection.
    :type image: OpenCV's image.
    :param ball_detection: ball_detection as a 5-dimensional tuple: (probability, x, y, width, height).
    :type ball_detection: 5-dimensional tuple.
    :param post1_detection: first goal post detection as a 5-dimensional tuple: (probability, x, y, width, height).
    :type post1_detection: 5-dimensional tuple.
    :param post2_detection: second goal post detection as a 5-dimensional tuple: (probability, x, y, width, height).
    :type post2_detection: 5-dimensional tuple.
    """
    if ball_detection[0] >= DETECTION_THRESHOLD:
        show_detection(image, ball_detection, BLUE, True)
    if post1_detection[0] >= DETECTION_THRESHOLD:
        show_detection(image, post1_detection, GREEN)
    if post2_detection[0] >= DETECTION_THRESHOLD:
        show_detection(image, post2_detection, RED)
    plt.figure()
    plt.imshow(image)
    plt.title(image_name)


# Creating the object detector
detector = YoloDetector('yolo_ball_goalpost')

# Iterating over images, running the detector and showing the detections
for i in range(1, NUM_IMAGES + 1):
    image_name = 'imagem' + str(i)
    image = load_image(image_name)
    # Converting color space since OpenCV loads images as BGR and our network as trained with RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tic = time.time()
    ball_detection, post1_detection, post2_detection = detector.detect(image)
    toc = time.time()
    print(image_name + ': [(ball_prob: ' + str(ball_detection[0]) + ', post1_prob: ' + str(post1_detection[0]) +
          ', post2_prob: ' + str(post2_detection[0]) + '), processing time: ' + str(toc - tic) + ']')
    show_detections(image_name, image, ball_detection, post1_detection, post2_detection)
    plt.savefig(image_name + '_detection.png')

plt.show()
