import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image
from feature import FeatureExtractor
from cost import Difference
from aggregation import CostAggregation
from computation import DisparityComputation
from refinement import DisparityRefinement


def read_image(filename):
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype('float32') / 127.5 - 1.0
    return np.expand_dims(image, -1)


class StereoNet:
    def __init__(self, height=1024, width=1024, channel=1, min_disp=-112.0, max_disp=16.0):
        self.height = height
        self.width = width
        self.channel = channel
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)
        self.model = None

    def build_model(self):
        # inputs
        left_image = keras.Input(shape=(self.height, self.width, self.channel))
        right_image = keras.Input(shape=(self.height, self.width, self.channel))

        # extract features
        extractor = FeatureExtractor(filters=32)
        left_feature = extractor(left_image)
        right_feature = extractor(right_image)

        # construct cost volume
        constructor = Difference(self.min_disp // 8, self.max_disp // 8)
        cost_volume = constructor([left_feature, right_feature])

        # cost aggregation
        aggregator = CostAggregation(filters=32)
        cost_volume = aggregator(cost_volume)

        # pre-refined disparity
        computer = DisparityComputation(self.min_disp // 8, self.max_disp // 8)
        d0 = computer(cost_volume)

        # hierarchical refinement
        refiner1 = DisparityRefinement(filters=32)  # [None, 256, 256, 1], 1/4 resolution
        left_image_4x = tf.image.resize(left_image, [self.height // 4, self.width // 4])
        d1 = refiner1([d0, left_image_4x])

        refiner2 = DisparityRefinement(filters=32)  # [None, 512, 512, 1], 1/2 resolution
        left_image_2x = tf.image.resize(left_image, [self.height // 2, self.width // 2])
        d2 = refiner2([d1, left_image_2x])

        refiner3 = DisparityRefinement(filters=32)  # [None, 1024, 1024, 1], full resolution
        d3 = refiner3([d2, left_image])

        # The predicted disparity map is always bilinearly up-sampled to match ground-truth resolution.
        d0 = tf.image.resize(d0, [self.height, self.width]) * 8
        d1 = tf.image.resize(d1, [self.height, self.width]) * 4
        d2 = tf.image.resize(d2, [self.height, self.width]) * 2

        # StereoNet model
        self.model = keras.Model(inputs=[left_image, right_image], outputs=[d0, d1, d2, d3])
        self.model.summary()

    def predict(self, left_dir, right_dir, output_dir, weights):
        self.model.load_weights(weights)
        lefts = os.listdir(left_dir)
        rights = os.listdir(right_dir)
        lefts.sort()
        rights.sort()
        assert len(lefts) == len(rights)
        for left, right in zip(lefts, rights):
            left_image = np.expand_dims(read_image(os.path.join(left_dir, left)), 0)
            right_image = np.expand_dims(read_image(os.path.join(right_dir, right)), 0)
            disparity = self.model.predict([left_image, right_image])  # [d0, d1, d2, d3]
            disparity = Image.fromarray(disparity[-1][0, :, :, 0])
            name = left.replace('_block_BW', '_disparityMap')
            disparity.save(os.path.join(output_dir, name))


if __name__ == '__main__':
    # testing
    left_dir = 'your directory path of left images'
    right_dir = 'your directory path of right images'
    output_dir = 'your directory path of predictions'
    weights = '../weights/StereoNet.h5'
    net = StereoNet()
    net.build_model()
    net.predict(left_dir, right_dir, output_dir, weights)
