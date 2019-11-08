from losses import *

import gc

glr = 1e-4
dlr = 1e-4
g1_optimizer = tf.keras.optimizers.RMSprop(glr, momentum=0.25)
g2_optimizer = tf.keras.optimizers.RMSprop(glr, momentum=0.25)
d1_optimizer = tf.keras.optimizers.RMSprop(dlr, momentum=0.25)
d2_optimizer = tf.keras.optimizers.RMSprop(dlr, momentum=0.25)

# Here, unlike the standard implementation, we do not cycle through both generators.
# We only require images similar to those that are pencil-drawn, not the other way around.
# The only cycle in this implementation is pencil -> regular -> pencil.
def train(genAB, genBA, discA, discB, X1, X2):
    real_A = X1
    real_B = X2
    with tf.GradientTape(persistent=True) as tape:
        fake_B = genAB(real_A)  # Sketch -> Color
        fake_A = genBA(real_B)  # Color -> Sketch

        cycled_A = genBA(fake_B)  # Sketch -> Color -> Sketch

        same_B = genAB(real_B)  
        same_A = genBA(real_A)  

        discA_fake_out = discA(fake_A)  
        discB_fake_out = discB(fake_B)

        discA_real_out = discA(real_A)
        discB_real_out = discB(real_B)

        g_loss = mae_criterion(discA_fake_out, tf.ones_like(discA_fake_out)) \
                 + mae_criterion(discB_fake_out, tf.ones_like(discB_fake_out)) \
                 + 10 * abs_criterion(real_A, cycled_A) \
                 + 5 * abs_criterion(real_A, same_A) \
                 + 5 * abs_criterion(real_B, same_B)

        discA_loss_real = mae_criterion(discA_real_out, tf.ones_like(discA_real_out))
        discB_loss_real = mae_criterion(discB_real_out, tf.ones_like(discB_real_out))

        discA_loss_fake = mae_criterion(discA_fake_out, tf.zeros_like(discA_fake_out))
        discB_loss_fake = mae_criterion(discB_fake_out, tf.zeros_like(discB_fake_out))

        discA_loss = (discA_loss_real + discA_loss_fake) / 2.0
        discB_loss = (discB_loss_real + discB_loss_fake) / 2.0

        d_loss = discA_loss + discB_loss

        net_loss = g_loss + d_loss

    G_vars = genAB.trainable_variables
    gradients = tape.gradient(g_loss, G_vars)
    grads_and_vars = zip(gradients, G_vars)
    clipped = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in grads_and_vars]
    g1_optimizer.apply_gradients(clipped)

    H_vars = genBA.trainable_variables
    gradients = tape.gradient(g_loss, H_vars)
    grads_and_vars = zip(gradients, H_vars)
    clipped = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in grads_and_vars]
    g1_optimizer.apply_gradients(clipped)

    D_vars = discA.trainable_variables
    gradients = tape.gradient(d_loss, D_vars)
    grads_and_vars = zip(gradients, D_vars)
    clipped = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in grads_and_vars]
    g1_optimizer.apply_gradients(clipped)

    E_vars = discB.trainable_variables
    gradients = tape.gradient(d_loss, E_vars)
    grads_and_vars = zip(gradients, E_vars)
    clipped = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in grads_and_vars]
    g1_optimizer.apply_gradients(clipped)

    del tape
    gc.collect()
    return net_loss, g_loss, d_loss
