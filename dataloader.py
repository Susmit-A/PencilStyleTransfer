import cv2
import numpy as np
import os

batch_size = 10


def get_image_list(path='images/'):
    images = {
        'pencil': [],
        'regular': []
    }

    data_pencil = os.listdir(path + 'pencil')
    for val in data_pencil:
        if val.endswith('.jpg'):
            images['pencil'].append(path+'pencil/'+val)

    data_regular = os.listdir(path + 'regular')
    for val in data_regular:
        if val.endswith('.jpg'):
            images['regular'].append(path + 'regular/' + val)

    return images


image_dataset = get_image_list()
len_pencil = len(image_dataset['pencil'])
len_regular = len(image_dataset['regular'])

data_X1 = []
data_X2 = []


def load_all_data():
    global data_X1, data_X2
    for img_path in image_dataset['pencil']:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        img = ((img / 127.5) - 1.0).astype(np.float32)
        data_X1.append(img)

    for img_path in image_dataset['regular']:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        img = ((img / 127.5) - 1.0).astype(np.float32)
        data_X2.append(img)

    data_X1 = np.asarray(data_X1)
    data_X2 = np.asarray(data_X2)


def create_batch(batch_size=batch_size):
    X1_ind = np.random.choice(len(data_X1), batch_size, replace=False)
    X2_ind = np.random.choice(len(data_X2), batch_size, replace=False)
    return np.stack(data_X1[X1_ind]), np.stack(data_X2[X2_ind])


def create_test_batch(bs=32):
    X1 = []
    X2 = []
    pencil = np.random.choice(image_dataset['pencil'], bs, replace=False)
    regular = np.random.choice(image_dataset['regular'], bs, replace=False)
    for img_path in pencil:
        img = cv2.imread(img_path)
        shp = np.shape(img[:-1])
        img = cv2.resize(img, (min(shp[0], 1024), min(shp[1], 1024)))
        img = ((img / 127.5) - 1.0).astype(np.float32)
        X1.append(img)

    for img_path in regular:
        img = cv2.imread(img_path)
        shp = np.shape(img[:-1])
        img = cv2.resize(img, (min(shp[0], 1024), min(shp[1], 1024)))
        img = ((img / 127.5) - 1.0).astype(np.float32)
        X2.append(img)

    return X1, X2
