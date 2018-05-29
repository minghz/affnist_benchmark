import os
import numpy as np
import scipy.io as spio
import tensorflow as tf

from PIL import Image
from config import cfg

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def load_centered(is_training):
    path = os.path.join('../mnist_data', 'mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // cfg.batch_size
        num_val_batch = 5000 // cfg.batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // cfg.batch_size
        return teX / 255., teY, num_te_batch


def load_transformed():
    # TODO build later
    print('Loading transformed images')


def get_batch(use_just_centered=True, is_training=True):
    if use_just_centered:
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_centered(is_training)
    else: # load transformed images
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_transformed(is_training)

    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.batch(data_queues,
                          batch_size=cfg.batch_size,
                          num_threads=1,
                          capacity=cfg.batch_size * 64,
                          enqueue_many=False,
                          allow_smaller_final_batch=False)
    return X, Y
