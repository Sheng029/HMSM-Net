import tensorflow as tf
from ubm import *


def conv3d(filters, kernel_size, strides, padding):
    return keras.layers.Conv3D(filters=filters, kernel_size=kernel_size,
                               strides=strides, padding=padding,
                               kernel_regularizer=keras.regularizers.l2(L2))


def conv3d_bn(filters, kernel_size, strides, padding, activation):
    conv = keras.layers.Conv3D(filters=filters, kernel_size=kernel_size,
                               strides=strides, padding=padding,
                               use_bias=False, kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(L2))
    bn = keras.layers.BatchNormalization()
    relu = keras.layers.ReLU()

    if activation:
        return keras.Sequential([conv, bn, relu])
    else:
        return keras.Sequential([conv, bn])


def trans_conv3d_bn(filters, kernel_size, strides, padding, activation):
    conv = keras.layers.Conv3DTranspose(filters=filters, kernel_size=kernel_size,
                                        strides=strides, padding=padding,
                                        use_bias=False, kernel_initializer='he_normal',
                                        kernel_regularizer=keras.regularizers.l2(L2))
    bn = keras.layers.BatchNormalization()
    relu = keras.layers.ReLU()

    if activation:
        return keras.Sequential([conv, bn, relu])
    else:
        return keras.Sequential([conv, bn])


def trans_conv3d(filters, kernel_size, strides, padding):
    return keras.layers.Conv3DTranspose(filters=filters, kernel_size=kernel_size,
                                        strides=strides, padding=padding,
                                        kernel_regularizer=keras.regularizers.l2(L2))


class Res3D(keras.Model):
    def __init__(self, filters):
        super(Res3D, self).__init__()
        self.conv1 = conv3d_bn(filters, 3, 1, 'same', True)
        self.conv2 = conv3d_bn(filters, 3, 1, 'same', False)
        self.relu = keras.layers.ReLU()

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x += inputs
        x = self.relu(x)
        return x


class CostVolume(keras.Model):
    def __init__(self, max_disp):
        super(CostVolume, self).__init__()
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


class NonLocal(keras.Model):
    def __init__(self, filters):
        super(NonLocal, self).__init__()
        self.conv_a = conv3d(filters // 2, 1, 1, 'valid')
        self.conv_b = conv3d(filters // 2, 1, 1, 'valid')
        self.conv_c = conv3d(filters // 2, 1, 1, 'valid')
        self.conv = conv3d(filters, 1, 1, 'valid')

    def call(self, inputs, training=None, mask=None):
        [B, D, H, W, C] = inputs.get_shape().as_list()
        a = self.conv_a(inputs)
        b = self.conv_b(inputs)
        c = self.conv_c(inputs)
        a = keras.layers.Reshape((D*H*W, C // 2))(a)   # [B, DHW, C//2]
        b = keras.layers.Reshape((D*H*W, C // 2))(b)   # [B, DHW, C//2]
        b = tf.transpose(b, [0, 2, 1])   # [B, C, DHW]
        ab = tf.matmul(a, b)   # [B, DHW, DHW]
        ab = tf.math.softmax(ab, -1)
        c = keras.layers.Reshape((D*H*W, C // 2))(c)   # [B, DHW, C//2]
        abc = tf.matmul(ab, c)   # [B, DHW, C//2]
        abc = keras.layers.Reshape((D, H, W, C // 2))(abc)
        x = self.conv(abc)
        x += inputs
        return x


class SoftArgMin(keras.Model):
    def __init__(self, max_disp):
        super(SoftArgMin, self).__init__()
        self.max_disp = max_disp

    def call(self, inputs, training=None, mask=None):
        assert inputs.shape[-1] == 2 * self.max_disp
        candidates = tf.linspace(-1.0 * self.max_disp, 1.0 * self.max_disp - 1.0, 2 * self.max_disp)
        prob_volume = tf.math.softmax(-1.0 * inputs, -1)
        disparity = tf.reduce_sum(candidates * prob_volume, -1, True)

        return disparity


class FFM(keras.Model):
    def __init__(self, filters, max_disp):
        super(FFM, self).__init__()
        self.trans2 = keras.Sequential([
            conv2d_bn(4 * filters, 1, 1, 'valid', 1, True),
            conv2d_bn(2 * filters, 3, 1, 'same', 1, True)])
        self.cost = CostVolume(max_disp=max_disp)
        self.conv3_0 = keras.Sequential([
            conv3d_bn(2 * filters, 3, 1, 'same', True),
            conv3d_bn(filters, 3, 1, 'same', True),
            Res3D(filters)])
        self.conv3_1 = keras.Sequential([
            conv3d_bn(2 * filters, 3, 2, 'same', True),
            Res3D(2 * filters)])
        self.conv3_2 = keras.Sequential([
            conv3d_bn(3 * filters, 3, 2, 'same', True),
            Res3D(3 * filters)])
        self.conv3_3 = keras.Sequential([
            conv3d_bn(4 * filters, 3, 2, 'same', True),
            Res3D(4 * filters)])
        self.conv3_4 = keras.Sequential([
            conv3d_bn(5 * filters, 3, 2, 'same', True),
            Res3D(5 * filters)])
        self.conv3_5 = keras.Sequential([
            NonLocal(5 * filters), NonLocal(5 * filters),
            NonLocal(5 * filters), NonLocal(5 * filters)])
        self.deconv3_3 = keras.Sequential([
            trans_conv3d_bn(4 * filters, 3, 2, 'same', True),
            Res3D(4 * filters)])
        self.deconv3_2 = keras.Sequential([
            trans_conv3d_bn(3 * filters, 3, 2, 'same', True),
            Res3D(3 * filters)])
        self.deconv3_1 = keras.Sequential([
            trans_conv3d_bn(2 * filters, 3, 2, 'same', True),
            Res3D(2 * filters)])
        self.deconv3_0 = keras.Sequential([
            trans_conv3d_bn(filters, 3, 2, 'same', True),
            Res3D(filters)])
        self.deconv3_4 = keras.Sequential([
            trans_conv3d_bn(filters // 2, 3, 2, 'same', True),
            trans_conv3d(1, 3, 2, 'same')])
        self.conv3_6 = conv3d(1, 3, 1, 'same')
        self.soft_arg_min = SoftArgMin(max_disp=max_disp * 4)

    def call(self, inputs, training=None, mask=None):
        # inputs: [left, right, dsp_att]
        assert len(inputs) == 3
        xl = self.trans2(inputs[0])
        xr = self.trans2(inputs[1])
        cost_volume = self.cost([xl, xr])
        cost_volume = self.conv3_0(cost_volume)
        c1 = self.conv3_1(cost_volume)
        c2 = self.conv3_2(c1)
        c3 = self.conv3_3(c2)
        c4 = self.conv3_4(c3)
        c = self.conv3_5(c4)
        dc3 = self.deconv3_3(c)
        dc3 += c3
        dc2 = self.deconv3_2(dc3)
        dc2 += c2
        dc1 = self.deconv3_1(dc2)
        dc1 += c1
        dc = self.deconv3_0(dc1)
        dc += cost_volume
        agg_cost = dc * inputs[2] + dc
        agg_cost = self.deconv3_4(agg_cost)
        agg_cost = self.conv3_6(agg_cost)   # [N, D, H, W, 1]
        agg_cost = tf.squeeze(agg_cost, -1)   # [N, D, H, W]
        agg_cost = tf.transpose(agg_cost, [0, 2, 3, 1])
        disp = self.soft_arg_min(agg_cost)
        return disp


# if __name__ == '__main__':
#     model = FFM(filters=32, max_disp=24)
#     model.build([(2, 256, 256, 512), (2, 256, 256, 512), (2, 48, 256, 256, 1)])
#     model.summary()
