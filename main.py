import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore")

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

from tensorflow.keras.utils import Progbar
from models import *
from dataloader import *
from training import *

from sklearn.utils import shuffle

import os

model_G = make_generator_resnet()  # First Generator
model_H = make_generator_resnet()  # Second Generator
model_D = make_discriminator()  # Discriminator for G
model_E = make_discriminator()  # Discriminator for H

if os.path.exists('model_D.h5'):
    print("Loading model_D")
    model_D.load_weights('model_D.h5')

if os.path.exists('model_E.h5'):
    print("Loading model_E")
    model_E.load_weights('model_E.h5')

if os.path.exists('model_G.h5'):
    print("Loading model_G")
    model_G.load_weights('model_G.h5')

if os.path.exists('model_H.h5'):
    print("Loading model_H")
    model_H.load_weights('model_H.h5')


print("Loading data")
load_all_data()
print("Data fetched into RAM")
print(np.shape(data_X1))


def train_loop(epochs=10, steps=1000, start_epoch=0):
    for epoch in range(start_epoch, start_epoch+epochs):
        image_dataset['pencil'] = shuffle(image_dataset['pencil'])
        image_dataset['regular'] = shuffle(image_dataset['regular'])
        print("Epoch ", epoch)
        steps = steps
        progressbar = Progbar(steps)
        gen_losses = []
        disc_losses = []

        for i in range(steps):
            X1, X2 = create_batch()
            loss, gl, dl = train(model_G, model_H, model_D, model_E, X1, X2)
            gen_losses.append(gl.numpy())
            disc_losses.append(dl.numpy())
            progressbar.update(i, [
                ('loss', loss.numpy()),
                ('generator loss', gen_losses[-1]),
                ('discriminator loss', disc_losses[-1])
            ])

            if i % 250 == 0:

                model_D.save('model_D.h5')
                model_E.save('model_E.h5')
                model_G.save('model_G.h5')
                model_H.save('model_H.h5')

                os.mkdir(str(epoch) + '_' + str(i))

                model_D.save(str(epoch) + '_' + str(i) + '/model_D.h5')
                model_E.save(str(epoch) + '_' + str(i) + '/model_E.h5')
                model_G.save(str(epoch) + '_' + str(i) + '/model_G.h5')
                model_H.save(str(epoch) + '_' + str(i) + '/model_H.h5')

                os.mkdir(str(epoch) + '_' + str(i) + '/sketch2color')
                os.mkdir(str(epoch) + '_' + str(i) + '/color2sketch')

                XY, XO = create_test_batch(8)
                j = 0
                for Xy in XY:
                    Xy = np.expand_dims(Xy, 0)
                    Yy = np.squeeze(np.around((1 + model_G.predict(Xy)) * 127.5))
                    Xy = np.squeeze(np.around((1 + Xy) * 127.5))

                    cv2.imwrite(str(epoch) + '_' + str(i) + '/sketch2color/' + str(j) + '.jpg', Yy)
                    cv2.imwrite(str(epoch) + '_' + str(i) + '/sketch2color/' + str(j) + '_actual.jpg', Xy)
                    j += 1

                j = 0
                for Xo in XO:
                    Xo = np.expand_dims(Xo, 0)
                    Yo = np.squeeze(np.around((1 + model_H.predict(Xo)) * 127.5))
                    Xo = np.squeeze(np.around((1 + Xo) * 127.5))
                    cv2.imwrite(str(epoch) + '_' + str(i) + '/color2sketch/' + str(j) + '.jpg', Yo)

                    cv2.imwrite(str(epoch) + '_' + str(i) + '/color2sketch/' + str(j) + '_actual.jpg', Xo)
                    j += 1

            gc.collect()


train_loop()
