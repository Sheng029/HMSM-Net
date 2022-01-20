import tensorflow as tf
import tensorflow.keras as keras


class CostConcatenation(keras.Model):
    def __init__(self, max_disp):
        super(CostConcatenation, self).__init__()
        self.max_disp = max_disp

    def call(self, inputs, training=None, mask=None):
        assert len(inputs) == 2

        cost_volume = []
        for i in range(-self.max_disp, self.max_disp):
            if i < 0:
                cost_volume.append(tf.pad(
                    tensor=tf.concat([inputs[0][:, :, :i, :], inputs[1][:, :, -i:, :]], -1),
                    paddings=[[0, 0], [0, 0], [0, -i], [0, 0]], mode='CONSTANT'))
            elif i > 0:
                cost_volume.append(tf.pad(
                    tensor=tf.concat([inputs[0][:, :, i:, :], inputs[1][:, :, :-i, :]], -1),
                    paddings=[[0, 0], [0, 0], [i, 0], [0, 0]], mode='CONSTANT'))
            else:
                cost_volume.append(tf.concat([inputs[0], inputs[1]], -1))
        cost_volume = tf.stack(cost_volume, 1)

        return cost_volume  # [N, D, H, W, C]


class CostDifference(keras.Model):
    def __init__(self, max_disp):
        super(CostDifference, self).__init__()
        self.max_disp = max_disp

    def call(self, inputs, training=None, mask=None):
        assert len(inputs) == 2

        cost_volume = []
        for i in range(-self.max_disp, self.max_disp):
            if i < 0:
                cost_volume.append(tf.pad(
                    tensor=inputs[0][:, :, :i, :] - inputs[1][:, :, -i:, :],
                    paddings=[[0, 0], [0, 0], [0, -i], [0, 0]], mode='CONSTANT'))
            elif i > 0:
                cost_volume.append(tf.pad(
                    tensor=inputs[0][:, :, i:, :] - inputs[1][:, :, :-i, :],
                    paddings=[[0, 0], [0, 0], [i, 0], [0, 0]], mode='CONSTANT'))
            else:
                cost_volume.append(inputs[0] - inputs[1])
        cost_volume = tf.stack(cost_volume, 1)

        return cost_volume  # [N, D, H, W, C]
