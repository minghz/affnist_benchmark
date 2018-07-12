import os
import sys
import numpy as np
import scipy.io as spio
import pdb #debugger

TOTAL_TRAINING_IMAGES = 60000
SAVE_DIR = 'peppered_training_and_validation_batches'


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


def validate_base_and_peppered(percent_centered, percent_peppered):
    percent_centered = int(filename.split('_')[0])
    percent_peppered = int(filename.split('_')[3])

    images_per_transformation = int((TOTAL_TRAINING_IMAGES * percent_peppered/100.0) / 32)
    num_img_peppered = images_per_transformation * 32

    num_img_base = int(TOTAL_TRAINING_IMAGES * percent_centered/100.0)

    data = loadmat(os.path.join(SAVE_DIR, filename))
    assert data['affNISTdata']['image'].shape == (1600, num_img_base + num_img_peppered)
    
    print(filename + ' OK. Centered: ' + str(num_img_base) + ' Peppered: ' + str(num_img_peppered) + ' Total: ' + str(num_img_base + num_img_peppered))


def validate(filename):
    print('Validating ' + filename, end='\r')
    filename_array = filename.split('_')
    validate_base_and_peppered(filename_array[0], filename_array[3])


if __name__ == '__main__':
    for filename in os.listdir(SAVE_DIR):
        if not filename == __file__ and not filename.startswith('.'):
            validate(filename)
