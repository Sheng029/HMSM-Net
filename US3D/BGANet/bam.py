import tensorflow as tf
from ubm import *


class GlobalContext(keras.Model):
    def __init__(self, filters, rate):
        super(GlobalContext, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=1, kernel_size=1, strides=1,
                                         kernel_regularizer=keras.regularizers.l2(L2))
        self.conv2 = keras.layers.Conv2D(filters=filters // rate, kernel_size=1, strides=1,
                                         kernel_regularizer=keras.regularizers.l2(L2))
        self.layer_norm = keras.layers.LayerNormalization()
        self.relu = keras.layers.ReLU()
        self.conv3 = keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1,
                                         kernel_regularizer=keras.regularizers.l2(L2))

    def call(self, inputs, training=None, mask=None):
        [b, h, w, c] = inputs.get_shape().as_list()
        input_x = keras.layers.Reshape((h*w, c))(inputs)
        input_x = tf.transpose(input_x, [0, 2, 1])

        context_mask = self.conv1(inputs)
        context_mask = keras.layers.Reshape((h*w, 1))(context_mask)
        context_mask = tf.math.softmax(context_mask, 1)

        context = tf.matmul(input_x, context_mask)
        context = tf.expand_dims(context, -1)
        context = tf.transpose(context, [0, 2, 3, 1])

        transform = self.conv2(context)
        transform = self.layer_norm(transform)
        transform = self.relu(transform)
        transform = self.conv3(transform)

        outputs = inputs + transform
        return outputs


class BAM(keras.Model):
    def __init__(self, filters, dsps):
        super(BAM, self).__init__()
        self.trans0 = FeatureTrans(4 * filters)
        self.conv1_0 = keras.Sequential(
            [GlobalContext(4 * filters, 4),
             BasicBlock(4 * filters, 1)])
        self.conv1_1 = keras.Sequential(
            [GlobalContext(4 * filters, 4),
             BasicBlock(4 * filters, 1)])
        self.conv1_2 = keras.Sequential(
            [GlobalContext(4 * filters, 4),
             BasicBlock(4 * filters, 1)])
        self.conv1_3 = keras.Sequential(
            [GlobalContext(4 * filters, 4),
             BasicBlock(4 * filters, 1)])
        self.cls_att = keras.Sequential(
            [conv2d_bn(2 * filters, 1, 1, 'valid', 1, True),
             conv2d(1, 1, 1, 'valid', 1),
             keras.layers.Activation('sigmoid')])
        self.dsp_att = keras.Sequential(
            [conv2d_bn(2 * filters, 1, 1, 'valid', 1, True),
             conv2d(2 * dsps, 1, 1, 'valid', 1),
             keras.layers.Activation('sigmoid')])

    def call(self, inputs, training=None, mask=None):
        x = self.trans0(inputs)
        x = self.conv1_0(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        cls = self.cls_att(x)
        dsp = self.dsp_att(x)
        dsp = tf.transpose(dsp, [0, 3, 1, 2])
        dsp = tf.expand_dims(dsp, -1)
        return [cls, dsp]


# if __name__ == '__main__':
#     model = BAM(filters=32, dsps=24)
#     model.build((2, 256, 256, 256))
#     model.summary()
