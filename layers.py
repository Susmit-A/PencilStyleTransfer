import tensorflow as tf
from tensorflow.keras.layers import *


class InstanceNormalization(Layer):
    def __init__(self, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        depth = input_shape[3]
        self.scale = self.add_weight(name="scale_" + str(depth),
                                     shape=[depth],
                                     initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32)
                                     )

        self.offset = self.add_weight(name="offset_" + str(depth),
                                      shape=[depth],
                                      initializer=tf.constant_initializer(0.0)
                                      )
        super(InstanceNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (inputs - mean) * inv
        return self.scale * normalized + self.offset


class Residual(Layer):
    def __init__(self, dim=256, ks=3, s=1, padding='VALID', stddev=0.02, from_list=None, **kwargs):
        self.dim = dim
        self.ks = ks
        self.s = s
        self.from_list = from_list
        self.padding = padding
        self.stddev = stddev
        super(Residual, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv2d1 = Conv2D(filters=self.dim, kernel_size=self.ks, strides=self.s, padding=self.padding,
                              activation=None,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=self.stddev),
                              bias_initializer=None)
        self.instnorm1 = InstanceNormalization()
        self.relu1 = ReLU()
        self.conv2d2 = Conv2D(filters=self.dim, kernel_size=self.ks, strides=self.s, padding=self.padding,
                              activation=None,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=self.stddev),
                              bias_initializer=None)
        self.instnorm2 = InstanceNormalization()

        super(Residual, self).build(input_shape)

    def call(self, x, **kwargs):
        p = int((self.ks - 1) / 2)
        y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")

        y = self.instnorm1(self.conv2d1(y))
        y = tf.pad(self.relu1(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = self.instnorm2(self.conv2d2(y))
        return y + x


class ResizeConv2D(Layer):
    def __init__(self, filters, kernel_size, resize_factor=2, strides=(1, 1), pads=(0, 0), activation=None, **kwargs):
        self.filters = filters
        self.ks = kernel_size
        self.strides = strides
        self.activation = activation
        self.pads = pads
        self.resize_factor = resize_factor
        super(ResizeConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv = Conv2D(self.filters, self.ks, self.strides, activation=self.activation, bias_initializer=None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    def call(self, inputs, **kwargs):
        inp = inputs
        inp_shp = tf.shape(inputs)
        x = tf.image.resize_images(inp, size=[inp_shp[1]*self.resize_factor, inp_shp[2]*self.resize_factor], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x = tf.pad(x, ([0, 0], [self.pads[0], self.pads[0]], [self.pads[1], self.pads[1]], [0, 0]), 'REFLECT')
        x = self.conv(x)
        return x
