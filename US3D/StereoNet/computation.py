import tensorflow as tf
import tensorflow.keras as keras


class DisparityComputation(keras.Model):
    """
    Compute disparity using Soft ArgMin.
    """
    def __init__(self, max_disp):
        super(DisparityComputation, self).__init__()
        self.max_disp = max_disp

    def call(self, inputs, training=None, mask=None):
        # inputs: [N, H, W, D], D = 2 * max_disp
        assert inputs.shape[-1] == 2 * self.max_disp

        disp_candidates = tf.linspace(-1.0 * self.max_disp, 1.0 * self.max_disp - 1.0, 2 * self.max_disp)
        prob_volume = tf.math.softmax(-1.0 * inputs, -1)
        disparity = tf.reduce_sum(disp_candidates * prob_volume, -1, True)

        return disparity     # [N, H, W, 1]
