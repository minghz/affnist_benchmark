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
    print('Loading centered images')

    if is_training:
        data_file = os.path.join(cfg.data_dir, 'just_centered', 'training_and_validation.mat')
        data = loadmat(data_file)
        images = data['affNISTdata']['image'].transpose().reshape(60000, 40, 40, 1).astype(np.float32)
        labels = data['affNISTdata']['label_int'].astype(np.uint8)
        assert images.shape == (60000, 40, 40, 1)
        assert labels.shape == (60000,)

        trX = images[:50000] / 255.
        trY = labels[:50000]

        valX = images[10000:, ] / 255.
        valY = labels[10000:]

        num_tr_batch = 50000 // cfg.batch_size
        num_val_batch = 10000 // cfg.batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch

    else:
        # NOTE: Swap those two lines below to get some basic transformed test
        data_file = os.path.join(cfg.data_dir, 'transformed', 'test_batches', '1.mat')
        #data_file = os.path.join(cfg.data_dir, 'just_centered', 'test.mat')
        data = loadmat(data_file)
        images = data['affNISTdata']['image'].transpose().reshape(10000, 40, 40, 1).astype(np.float32)
        labels = data['affNISTdata']['label_int'].astype(np.float32)
        assert images.shape == (10000, 40, 40, 1)
        assert labels.shape == (10000,)

        imgs = images / 255.
        labs = labels
        num_te_batch = 10000 // cfg.batch_size

        return imgs, labs, num_te_batch


def load_transformed():
    # TODO build later
    print('Loading transformed images')


def get_batch(use_just_centered=True, is_training=True):
    if use_just_centered:
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_centered(is_training)
    else: # load transformed images
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_transformed(is_training)

    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues,
                                  num_threads=8,
                                  batch_size=cfg.batch_size,
                                  capacity=cfg.batch_size * 64,
                                  min_after_dequeue=cfg.batch_size * 32,
                                  allow_smaller_final_batch=False)
    return X, Y
