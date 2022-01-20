import os
import cv2
import numpy as np
import tensorflow.keras as keras
from PIL import Image


def read_image(filename):
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype('float32') / 127.5 - 1.0
    return np.expand_dims(image, -1)


class DenseMapNet:
    def __init__(self, height=1024, width=1024, channel=1, min_disp=-112.0, max_disp=16.0):
        self.height = height
        self.width = width
        self.channel = channel
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.model = None

    def build_model(self, dropout=0.2):
        left = keras.Input(shape=(self.height, self.width, self.channel))
        right = keras.Input(shape=(self.height, self.width, self.channel))

        # left image as reference
        x = keras.layers.Conv2D(filters=16, kernel_size=5, padding='same')(left)
        xleft = keras.layers.Conv2D(filters=1, kernel_size=5, padding='same', dilation_rate=2)(left)

        # left and right images for disparity estimation
        xin = keras.layers.Concatenate()([left, right])
        xin = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same')(xin)

        # image reduced by 8
        x8 = keras.layers.MaxPooling2D(8)(xin)
        x8 = keras.layers.BatchNormalization()(x8)
        x8 = keras.layers.Activation('relu')(x8)

        dilation_rate = 1
        y = x8
        # correspondence network
        # parallel cnn at increasing dilation rate
        for i in range(4):
            a = keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', dilation_rate=dilation_rate)(x8)
            a = keras.layers.Dropout(dropout)(a)
            y = keras.layers.Concatenate()([a, y])
            dilation_rate += 1

        dilation_rate = 1
        x = keras.layers.MaxPooling2D(8)(x)
        # disparity network
        # dense interconnection inspired by DenseNet
        for i in range(4):
            x = keras.layers.Concatenate()([x, y])
            y = keras.layers.BatchNormalization()(x)
            y = keras.layers.Activation('relu')(y)
            y = keras.layers.Conv2D(filters=64, kernel_size=1, padding='same')(y)

            y = keras.layers.BatchNormalization()(y)
            y = keras.layers.Activation('relu')(y)
            y = keras.layers.Conv2D(filters=16, kernel_size=5, padding='same', dilation_rate=dilation_rate)(y)
            y = keras.layers.Dropout(dropout)(y)
            dilation_rate += 1

        # disparity estimate scaled back to original image size
        x = keras.layers.Concatenate()([x, y])
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(filters=32, kernel_size=1, padding='same')(x)
        x = keras.layers.UpSampling2D(8)(x)

        # left image skip connection to disparity estimate
        x = keras.layers.Concatenate()([x, xleft])
        y = keras.layers.BatchNormalization()(x)
        y = keras.layers.Activation('relu')(y)
        y = keras.layers.Conv2D(filters=16, kernel_size=5, padding='same')(y)

        x = keras.layers.Concatenate()([x, y])
        y = keras.layers.BatchNormalization()(x)
        y = keras.layers.Activation('relu')(y)
        yout = keras.layers.Conv2DTranspose(filters=1, kernel_size=9, padding='same')(y)

        # densemapnet model
        self.model = keras.Model(inputs=[left, right], outputs=yout)
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
            disparity = Image.fromarray(disparity[0, :, :, 0])
            name = left.replace('_block_BW', '_disparityMap')
            disparity.save(os.path.join(output_dir, name))


if __name__ == '__main__':
    # testing
    left_dir = 'your directory path of left images'
    right_dir = 'your directory path of right images'
    output_dir = 'your directory path of predictions'
    weights = '../weights/DenseMapNet.h5'
    net = DenseMapNet()
    net.build_model()
    net.predict(left_dir, right_dir, output_dir, weights)
