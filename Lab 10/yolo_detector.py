from keras.models import load_model
import cv2
import numpy as np
from utils import sigmoid


class YoloDetector:
    """
    Represents an object detector for robot soccer based on the YOLO algorithm.
    """
    def __init__(self, model_name, anchor_box_ball=(5, 5), anchor_box_post=(2, 5)):
        """
        Constructs an object detector for robot soccer based on the YOLO algorithm.

        :param model_name: name of the neural network model which will be loaded.
        :type model_name: str.
        :param anchor_box_ball: dimensions of the anchor box used for the ball.
        :type anchor_box_ball: bidimensional tuple.
        :param anchor_box_post: dimensions of the anchor box used for the goal post.
        :type anchor_box_post: bidimensional tuple.
        """
        self.network = load_model(model_name + '.hdf5')
        self.network.summary()  # prints the neural network summary
        self.anchor_box_ball = anchor_box_ball
        self.anchor_box_post = anchor_box_post

    def detect(self, image):
        """
        Detects robot soccer's objects given the robot's camera image.

        :param image: image from the robot camera in 640x480 resolution and RGB color space.
        :type image: OpenCV's image.
        :return: (ball_detection, post1_detection, post2_detection), where each detection is given
                by a 5-dimensional tuple: (probability, x, y, width, height).
        :rtype: 3-dimensional tuple of 5-dimensional tuples.
        """
        # Todo: implement object detection logic
        image = self.preprocess_image(image)
        output = self.network.predict(image)
        [ball_detection, post1_detection, post2_detection] = self.process_yolo_output(output)
        return ball_detection, post1_detection, post2_detection

    def preprocess_image(self, image):
        """
        Preprocesses the camera image to adapt it to the neural network.

        :param image: image from the robot camera in 640x480 resolution and RGB color space.
        :type image: OpenCV's image.
        :return: image suitable for use in the neural network.
        :rtype: NumPy 4-dimensional array with dimensions (1, 120, 160, 3).
        """
        # Todo: implement image preprocessing logic
        image = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
        image = image/255
        image = np.array(image)
        image = np.reshape(image, (1, 120, 160, 3))
        return image

    def process_yolo_output(self, output):
        """
        Processes the neural network's output to yield the detections.

        :param output: neural network's output.
        :type output: NumPy 4-dimensional array with dimensions (1, 15, 20, 10).
        :return: (ball_detection, post1_detection, post2_detection), where each detection is given
                by a 5-dimensional tuple: (probability, x, y, width, height).
        :rtype: 3-dimensional tuple of 5-dimensional tuples.
        """
        coord_scale = 4 * 8  # coordinate scale used for computing the x and y coordinates of the BB's center
        bb_scale = 640  # bounding box scale used for computing width and height
        output = np.reshape(output, (15, 20, 10))  # reshaping to remove the first dimension
        # Todo: implement YOLO logic
        ib = 0
        jb = 0
        ip1 = 0
        jp1 = 0
        ip2 = 0
        jp2 = 0
        for i in range(15):
            for j in range(20):
                if output[i][j][0] >= output[ib][jb][0]:
                    ib = i
                    jb = j
                if output[i][j][5] >= output[ip2][jp2][5]:
                    if output[i][j][5] >= output[ip1][jp1][5]:
                        ip2 = ip1
                        jp2 = jp1
                        ip1 = i
                        jp1 = j
                    else:
                        ip2 = i
                        jp2 = j
        ball_detection = (sigmoid(output[ib][jb][0]), (jb + sigmoid(output[ib][jb][1]))*coord_scale, (ib + sigmoid(output[ib][jb][2]))*coord_scale, bb_scale*self.anchor_box_ball[0]*np.exp(output[ib][jb][3]), bb_scale*self.anchor_box_ball[1]*np.exp(output[ib][jb][4]))  # Todo: change this line
        post1_detection = (sigmoid(output[ip1][jp1][5]), (jp1 + sigmoid(output[ip1][jp1][6]))*coord_scale, (ip1 + sigmoid(output[ip1][jp1][7]))*coord_scale, bb_scale*self.anchor_box_post[0]*np.exp(output[ip1][jp1][8]), bb_scale*self.anchor_box_post[1]*np.exp(output[ip1][jp1][9]))  # Todo: change this line
        post2_detection = (sigmoid(output[ip2][jp2][5]), (jp2 + sigmoid(output[ip2][jp2][6]))*coord_scale, (ip2 + sigmoid(output[ip2][jp2][7]))*coord_scale, bb_scale*self.anchor_box_post[0]*np.exp(output[ip2][jp2][8]), bb_scale*self.anchor_box_post[1]*np.exp(output[ip2][jp2][9]))  # Todo: change this line
        return ball_detection, post1_detection, post2_detection
