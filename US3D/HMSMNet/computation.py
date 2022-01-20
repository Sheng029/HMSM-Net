import tensorflow as tf
import tensorflow.keras as keras


class Estimation(keras.Model):
    def __init__(self, max_disp):
        super(Estimation, self).__init__()
        self.max_disp = max_disp
        self.conv = keras.layers.Conv3D(filters=1, kernel_size=3,
                                        strides=1, padding='same',
                                        kernel_initializer='he_normal',
                                        kernel_regularizer=keras.regularizers.l2(1.0e-5))

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)     # [N, D, H, W, 1]
        x = tf.squeeze(x, -1)     # [N, D, H, W]
        x = tf.transpose(x, (0, 2, 3, 1))     # [N, H, W, D]

        assert x.shape[-1] == 2 * self.max_disp
        disp_candidates = tf.linspace(-1.0 * self.max_disp, 1.0 * self.max_disp - 1.0, 2 * self.max_disp)
        prob_volume = tf.math.softmax(-1.0 * x, -1)
        disparity = tf.reduce_sum(disp_candidates * prob_volume, -1, True)
        return disparity
