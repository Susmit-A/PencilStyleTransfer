from tensorflow.keras import Model
from layers import *


def make_discriminator():
    inp = Input((None, None, 3))

    h0 = Conv2D(32, (4, 4), bias_initializer=None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), strides=2)(inp)
    h0 = LeakyReLU()(h0)
    h1 = Conv2D(64, (4, 4), bias_initializer=None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), strides=2)(h0)
    h1 = BatchNormalization()(h1)
    h1 = LeakyReLU()(h1)
    h2 = Conv2D(128, (4, 4), bias_initializer=None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), strides=2)(h1)
    h2 = BatchNormalization()(h2)
    h2 = LeakyReLU()(h2)
    h3 = Conv2D(256, (4, 4), bias_initializer=None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), strides=2)(h2)
    h3 = BatchNormalization()(h3)
    h3 = LeakyReLU()(h3)
    h4 = Conv2D(1, (3, 3), bias_initializer=None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))(h3)
    m = Model(inputs=[inp], outputs=[h4])
    return m


def make_conv():
    inp = Input((None, None, 3))
    x = Conv2D(32, (4, 4), bias_initializer=None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), strides=2)(inp)
    x = InstanceNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, (4, 4), bias_initializer=None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), strides=2)(x)
    x = InstanceNormalization()(x)
    x = ReLU()(x)
    return Model(inputs=[inp], outputs=[x])


def make_deconv():
    inp = Input((None, None, 64))
    x = ResizeConv2D(64, (4, 4), pads=(3, 3))(inp)
    x = InstanceNormalization()(x)
    x = ReLU()(x)
    x = ResizeConv2D(32, (4, 4), pads=(3, 3))(x)
    x = InstanceNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(3, (2, 2))(x)
    x = Activation('tanh')(x)
    return Model(inputs=[inp], outputs=[x])


def make_slide(channels):

    x = Input((None, None, channels))
    r1 = Residual(dim=channels)(x)
    r2 = Residual(dim=channels)(r1)
    r3 = Residual(dim=channels)(r2)
    r4 = Residual(dim=channels)(r3)
    r5 = Residual(dim=channels)(r4)
    r6 = Residual(dim=channels)(r5)
    r7 = Residual(dim=channels)(r6)
    model = Model(inputs=[x], outputs=[r7])
    return model


def make_generator_resnet():
    inp = Input((None, None, 3))

    conv = make_conv()
    deconv = make_deconv()

    x = conv(inp)
    res = make_slide(64)
    res_out = res(x)
    out = deconv(res_out)

    m = Model(inputs=[inp], outputs=[out])
    return m
