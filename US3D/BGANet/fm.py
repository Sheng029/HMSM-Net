from ubm import *


class Fusion(keras.Model):
    def __init__(self, filters, num_class):
        super(Fusion, self).__init__()
        # segmentation
        self.conv4_0 = conv2d_bn(filters, 3, 1, 'same', 1, True)
        self.conv4_1 = conv2d_bn(2 * filters, 3, 2, 'same', 1, True)
        self.conv4_2 = conv2d_bn(3 * filters, 3, 2, 'same', 1, True)
        self.conv4_3 = conv2d_bn(4 * filters, 3, 2, 'same', 1, True)
        self.conv4_4 = conv2d_bn(5 * filters, 3, 2, 'same', 1, True)
        self.deconv4_3 = trans_conv2d_bn(4 * filters, 3, 2, 'same', True)
        self.deconv4_2 = trans_conv2d_bn(3 * filters, 3, 2, 'same', True)
        self.deconv4_1 = trans_conv2d_bn(2 * filters, 3, 2, 'same', True)
        self.deconv4_0 = trans_conv2d_bn(filters, 3, 2, 'same', True)
        self.conv4_5 = conv2d(num_class, 3, 1, 'same', 1)
        # disparity
        self.conv5_0 = conv2d_bn(filters, 3, 1, 'same', 1, True)
        self.conv5_1 = conv2d_bn(2 * filters, 3, 2, 'same', 1, True)
        self.conv5_2 = conv2d_bn(3 * filters, 3, 2, 'same', 1, True)
        self.conv5_3 = conv2d_bn(4 * filters, 3, 2, 'same', 1, True)
        self.conv5_4 = conv2d_bn(5 * filters, 3, 2, 'same', 1, True)
        self.deconv5_3 = trans_conv2d_bn(4 * filters, 3, 2, 'same', True)
        self.deconv5_2 = trans_conv2d_bn(3 * filters, 3, 2, 'same', True)
        self.deconv5_1 = trans_conv2d_bn(2 * filters, 3, 2, 'same', True)
        self.deconv5_0 = trans_conv2d_bn(filters, 3, 2, 'same', True)
        self.conv5_5 = conv2d(1, 3, 1, 'same', 1)

    def call(self, inputs, training=None, mask=None):
        # inputs: [[left_img, dsp, score_map], [left_img, dsp, seg]]
        assert len(inputs) == 2
        # segmentation
        x = self.conv4_0(inputs[0])
        x1 = self.conv4_1(x)
        x2 = self.conv4_2(x1)
        x3 = self.conv4_3(x2)
        x4 = self.conv4_4(x3)
        dx3 = self.deconv4_3(x4)
        dx3 += x3
        dx2 = self.deconv4_2(dx3)
        dx2 += x2
        dx1 = self.deconv4_1(dx2)
        dx1 += x1
        dx = self.deconv4_0(dx1)
        dx += x
        dx = self.conv4_5(dx)
        # disparity
        y = self.conv5_0(inputs[0])
        y1 = self.conv5_1(y)
        y2 = self.conv5_2(y1)
        y3 = self.conv5_3(y2)
        y4 = self.conv5_4(y3)
        dy3 = self.deconv5_3(y4)
        dy3 += y3
        dy2 = self.deconv5_2(dy3)
        dy2 += y2
        dy1 = self.deconv5_1(dy2)
        dy1 += y1
        dy = self.deconv5_0(dy1)
        dy += y
        dy = self.conv5_5(dy)
        return [dx, dy]


# if __name__ == '__main__':
#     model = Fusion(filters=32, num_class=6)
#     model.build([(2, 1024, 1024, 10), (2, 1024, 1024, 5)])
#     model.summary()
