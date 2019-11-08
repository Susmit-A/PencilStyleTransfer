import tensorflow as tf

identity_loss_fn = tf.keras.losses.MeanAbsoluteError()


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def identity_loss(real, same):
    return identity_loss_fn(real, same)
