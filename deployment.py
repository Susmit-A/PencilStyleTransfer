import tensorflow as tf
config = tf.ConfigProto(device_count={'GPU': 0})
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

from models import *

import cv2
import numpy as np
import sys

model = make_generator_resnet()
model.load_weights('model_H.h5')

img = cv2.imread(sys.argv[1])
img = (img / 127.5) - 1.0
img = np.expand_dims(img, 0)

out = np.squeeze(model.predict(img))
out = np.around((out + 1.0)*127.5)

cv2.imwrite('out.jpg', out)

