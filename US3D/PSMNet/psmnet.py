import os
import cv2
import numpy as np
from PIL import Image
from component import *


def read_image(filename):
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype('float32') / 127.5 - 1.0
    return image


class PSMNet:
    def __init__(self, height=1024, width=1024, channel=3, max_disp=96):
        self.height = height
        self.width = width
        self.channel = channel
        self.max_disp = max_disp
        self.model = None

    def build_model(self):
        # inputs
        left_image = keras.Input(shape=(self.height, self.width, self.channel))
        right_image = keras.Input(shape=(self.height, self.width, self.channel))

        # CNN + SPP module
        extractor = FeatureExtractor(filters=32)
        left_feature = extractor(left_image)
        right_feature = extractor(right_image)

        # cost volume
        constructor = CostVolume(max_disp=self.max_disp // 4)
        cost_volume = constructor([left_feature, right_feature])

        # 3D CNN (stacked hourglass)
        hourglass = StackedHourglass(filters=32)
        [out1, out2, out3] = hourglass(cost_volume)

        # disparity
        estimation = Estimation(max_disp=self.max_disp)
        d1 = estimation(out1)
        d2 = estimation(out2)
        d3 = estimation(out3)

        # build model
        self.model = keras.Model(inputs=[left_image, right_image], outputs=[d1, d2, d3])
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
            disparity = self.model.predict([left_image, right_image])  # [d1, d2, d3]
            disparity = Image.fromarray(disparity[-1][0, :, :, 0])
            name = left.replace('RGB', 'DSP')
            disparity.save(os.path.join(output_dir, name))


if __name__ == '__main__':
    # testing
    left_dir = 'your directory path of left images'
    right_dir = 'your directory path of right images'
    output_dir = 'your directory path of predictions'
    weights = '../weights/PSMNet.h5'
    net = PSMNet()
    net.build_model()
    net.predict(left_dir, right_dir, output_dir, weights)
