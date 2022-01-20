import tensorflow.keras as keras


L2 = 1.0e-5


def conv2d(filters, kernel_size, strides, padding, dilation_rate):
    return keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                               strides=strides, padding=padding,
                               dilation_rate=dilation_rate, use_bias=True,
                               kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(L2))


def conv2d_bn(filters, kernel_size, strides, padding, dilation_rate, activation):
    conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                               strides=strides, padding=padding,
                               dilation_rate=dilation_rate, use_bias=False,
                               kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(L2))
    bn = keras.layers.BatchNormalization()
    relu = keras.layers.ReLU()

    if activation:
        return keras.Sequential([conv, bn, relu])
    else:
        return keras.Sequential([conv, bn])


def trans_conv2d_bn(filters, kernel_size, strides, padding, activation):
    conv = keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                        padding=padding, use_bias=False, kernel_initializer='he_normal',
                                        kernel_regularizer=keras.regularizers.l2(L2))
    bn = keras.layers.BatchNormalization()
    relu = keras.layers.ReLU()

    if activation:
        return keras.Sequential([conv, bn, relu])
    else:
        return keras.Sequential([conv, bn])


def trans_conv2d(filters, kernel_size, strides, padding):
    return keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                        padding=padding, kernel_initializer='he_normal',
                                        kernel_regularizer=keras.regularizers.l2(L2))


class BasicBlock(keras.Model):
    def __init__(self, filters, dilation_rate):
        super(BasicBlock, self).__init__()
        self.conv1 = conv2d_bn(filters, 3, 1, 'same', dilation_rate, True)
        self.conv2 = conv2d_bn(filters, 3, 1, 'same', dilation_rate, False)
        self.relu = keras.layers.ReLU()

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x += inputs
        x = self.relu(x)
        return x


class IdentityBlock(keras.Model):
    def __init__(self, filters):
        super(IdentityBlock, self).__init__()
        self.conv1 = conv2d_bn(filters, 1, 1, 'valid', 1, True)
        self.conv2 = conv2d_bn(filters, 3, 1, 'same', 1, True)
        self.conv3 = conv2d_bn(2 * filters, 1, 1, 'same', 1, False)
        self.relu = keras.layers.ReLU()

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x += inputs
        x = self.relu(x)
        return x


class FeatureTrans(keras.Model):
    def __init__(self, filters):
        super(FeatureTrans, self).__init__()
        self.conv1 = conv2d_bn(2 * filters, 1, 1, 'valid', 1, True)
        self.conv2 = conv2d_bn(filters, 1, 1, 'valid', 1, True)
        self.conv3 = BasicBlock(filters, 1)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


def make_basic_blocks(filters, dilation_rate, num):
    blocks = keras.Sequential()
    for i in range(num):
        blocks.add(BasicBlock(filters, dilation_rate))
    return blocks


def make_identity_blocks(filters, num):
    blocks = keras.Sequential()
    for i in range(num):
        blocks.add(IdentityBlock(filters))
    return blocks


class ASPP(keras.Model):
    def __init__(self, filters):
        super(ASPP, self).__init__()
        self.conv0 = conv2d_bn(filters, 1, 1, 'valid', 1, True)
        self.conv1 = conv2d_bn(filters, 3, 1, 'same', 6, True)
        self.conv2 = conv2d_bn(filters, 3, 1, 'same', 12, True)
        self.conv3 = conv2d_bn(filters, 3, 1, 'same', 18, True)

    def call(self, inputs, training=None, mask=None):
        x0 = self.conv0(inputs)
        x1 = self.conv1(inputs)
        x2 = self.conv2(inputs)
        x3 = self.conv3(inputs)
        return [x0, x1, x2, x3]


class UBM(keras.Model):
    def __init__(self, filters):
        super(UBM, self).__init__()
        self.conv0_1 = conv2d_bn(filters, 3, 2, 'same', 1, True)
        self.conv0_2 = keras.Sequential(
            [conv2d_bn(filters, 3, 1, 'same', 1, True),
             conv2d_bn(filters, 3, 1, 'same', 1, True),
             conv2d_bn(2 * filters, 3, 1, 'same', 1, True)])
        self.conv0_3 = make_basic_blocks(2 * filters, 1, 3)
        self.conv0_4 = keras.Sequential(
            [conv2d_bn(4 * filters, 3, 2, 'same', 1, True),
             make_identity_blocks(2 * filters, 16)])
        self.conv0_5 = keras.Sequential(
            [conv2d_bn(6 * filters, 3, 1, 'same', 1, True),
             BasicBlock(6 * filters, 2)])
        self.conv0_6 = keras.Sequential(
            [conv2d_bn(8 * filters, 3, 1, 'same', 1, True),
             BasicBlock(8 * filters, 4)])
        self.aspp = ASPP(2 * filters)
        self.concat = keras.layers.Concatenate()
        self.conv0_7 = keras.Sequential(
            [conv2d_bn(24 * filters, 3, 1, 'same', 1, True),
             conv2d(16 * filters, 3, 1, 'same', 1)])

    def call(self, inputs, training=None, mask=None):
        x0 = self.conv0_1(inputs)
        x0 = self.conv0_2(x0)
        x0 = self.conv0_3(x0)
        x0 = self.conv0_4(x0)
        x1 = self.conv0_5(x0)
        x2 = self.conv0_6(x1)
        [a0, a1, a2, a3] = self.aspp(x2)
        x = self.concat([x0, x1, x2, a0, a1, a2, a3])
        x = self.conv0_7(x)
        return x


# if __name__ == '__main__':
#     model = UBM(filters=32)
#     model.build((2, 1024, 1024, 3))
#     model.summary()
