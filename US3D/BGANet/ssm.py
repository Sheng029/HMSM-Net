from ubm import *


class SPP(keras.Model):
    def __init__(self, filters):
        super(SPP, self).__init__()
        self.branch0 = keras.Sequential([
            keras.layers.AvgPool2D((64, 64)),
            conv2d_bn(filters, 1, 1, 'valid', 1, True),
            keras.layers.UpSampling2D(size=(64, 64), interpolation='bilinear')])
        self.branch1 = keras.Sequential([
            keras.layers.AvgPool2D((32, 32)),
            conv2d_bn(filters, 1, 1, 'valid', 1, True),
            keras.layers.UpSampling2D(size=(32, 32), interpolation='bilinear')])
        self.branch2 = keras.Sequential([
            keras.layers.AvgPool2D((16, 16)),
            conv2d_bn(filters, 1, 1, 'valid', 1, True),
            keras.layers.UpSampling2D(size=(16, 16), interpolation='bilinear')])
        self.branch3 = keras.Sequential([
            keras.layers.AvgPool2D((8, 8)),
            conv2d_bn(filters, 1, 1, 'valid', 1, True),
            keras.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')])

    def call(self, inputs, training=None, mask=None):
        x0 = self.branch0(inputs)
        x1 = self.branch1(inputs)
        x2 = self.branch2(inputs)
        x3 = self.branch3(inputs)
        return [x0, x1, x2, x3]


class SSM(keras.Model):
    def __init__(self, filters, num_class):
        super(SSM, self).__init__()
        self.trans1 = FeatureTrans(4 * filters)
        self.conv2_0 = make_identity_blocks(2 * filters, 16)
        self.spp = SPP(filters)
        self.concat = keras.layers.Concatenate()
        self.conv2_1 = keras.Sequential([
            conv2d_bn(4 * filters, 3, 1, 'same', 1, True),
            conv2d_bn(2 * filters, 3, 1, 'same', 1, True)])
        self.deconv2_0 = keras.Sequential([
            trans_conv2d_bn(filters, 3, 2, 'same', True),
            trans_conv2d(num_class, 3, 2, 'same')])
        self.conv2_2 = conv2d(num_class, 3, 1, 'same', 1)
        self.softmax = keras.layers.Softmax()

    def call(self, inputs, training=None, mask=None):
        assert len(inputs) == 2
        x = self.trans1(inputs[0])
        x = self.conv2_0(x)
        [x0, x1, x2, x3] = self.spp(x)
        x = self.concat([x, x0, x1, x2, x3])
        x = self.conv2_1(x)
        x = x * inputs[1] + x
        x = self.deconv2_0(x)
        x = self.conv2_2(x)
        x = self.softmax(x)
        return x


# if __name__ == '__main__':
#     model = SSM(filters=32, num_class=6)
#     model.build([(2, 256, 256, 128), (2, 256, 256, 1)])
#     model.summary()
