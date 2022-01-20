import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image
from ubm import UBM
from bam import BAM
from ssm import SSM
from ffm import FFM
from fm import Fusion


def read_image(filename):
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype('float32') / 127.5 - 1.0
    return image


class BGANet:
    def __init__(self, height=1024, width=1024, channel=3, max_disp=96, num_class=6, filters=16):
        self.height = height
        self.width = width
        self.channel = channel
        self.max_disp = max_disp
        self.num_class = num_class
        self.filters = filters
        self.model = None

    def build_model(self):
        # inputs
        left_image = keras.Input((self.height, self.width, self.channel))
        right_image = keras.Input((self.height, self.width, self.channel))

        # unified backbone module
        backbone = UBM(filters=self.filters)
        left = backbone(left_image)
        right = backbone(right_image)

        # bidirectional guided attention module
        attention = BAM(filters=self.filters, dsps=self.max_disp // 4)
        [cls_att, dsp_att] = attention(left)

        # semantic segmentation module
        segmentation = SSM(filters=self.filters, num_class=self.num_class)
        init_score_map = segmentation([left, cls_att])
        init_seg_map = tf.argmax(init_score_map, -1)
        init_seg_map = tf.cast(init_seg_map, tf.float32)
        init_seg_map = tf.expand_dims(init_seg_map, -1)

        # feature matching module
        matching = FFM(filters=self.filters, max_disp=self.max_disp // 4)
        init_dsp_map = matching([left, right, dsp_att])

        seg_concat = tf.concat([left_image, init_dsp_map, init_score_map], -1)
        dsp_concat = tf.concat([left_image, init_dsp_map, init_seg_map], -1)

        # bidirectional fusion module
        fusion = Fusion(filters=self.filters, num_class=self.num_class)
        [seg_residual, dsp_residual] = fusion([seg_concat, dsp_concat])

        score_map = init_score_map + seg_residual
        score_map = tf.math.softmax(score_map, -1)
        dsp_map = init_dsp_map + dsp_residual

        self.model = keras.Model(inputs=[left_image, right_image],
                                 outputs=[init_score_map, score_map, init_dsp_map, dsp_map])
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
            disparity = self.model.predict([left_image, right_image])
            disparity = disparity[-1][0, :, :, 0]
            disparity = Image.fromarray(disparity)
            name = left.replace('RGB', 'DSP')
            disparity.save(os.path.join(output_dir, name))


if __name__ == '__main__':
    # testing
    left_dir = 'your directory path of left images'
    right_dir = 'your directory path of right images'
    output_dir = 'your directory path of predictions'
    weights = '../weights/BGANet.h5'
    net = BGANet(1024, 1024, 3, 96, 6, 16)
    net.build_model()
    net.predict(left_dir, right_dir, output_dir, weights)
