import os
import sys
import numpy as np
import scipy.io as spio

# Downloading data
# TODO: One day automate downloading input process
HOMEPAGE = "https://www.cs.toronto.edu/~tijmen/affNIST/32x/"
JUST_CENTERED_TEST_URL = HOMEPAGE + "just_centered/test.mat.zip"
JUST_CENTERED_TRAINING_AND_VALIDATION_URL = HOMEPAGE + "just_centered/training_and_validation.mat.zip"
TRANSFORMED_TEST_BATCHES_URL = HOMEPAGE + "transformed/test_batches.zip"
TRANSFORMED_TRAINING_AND_VALIDATION_BATCHES_URL = HOMEPAGE + "transformed/training_and_validation_batches.zip"

# Generate input
TOTAL_TRAINING_IMAGES = 60000
CENTERED_IMG_DIR = 'just_centered'
TRANSFORMED_TRAINING_IMG_DIR = 'transformed/training_and_validation_batches'
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


def centered_input_dict():
    data_file = os.path.join(CENTERED_IMG_DIR, 'training_and_validation.mat')
    data = loadmat(data_file)
    return {'affNISTdata': {'image': data['affNISTdata']['image'],
                            'label_int': data['affNISTdata']['label_int']}}


def generate_peppered(percentage_of_centered_images, percentage_of_transformed_images):
    check_output_dir(percentage_of_centered_images, percentage_of_transformed_images)

    images_per_transformation = int((TOTAL_TRAINING_IMAGES * percentage_of_transformed_images/100.0) / 32)
    num_img_to_pepper = images_per_transformation * 32
    num_img_peppered = 0

    peppered = centered_input_dict()

    for t in range(1, 33):
        data_file = os.path.join(TRANSFORMED_TRAINING_IMG_DIR, str(t) + '.mat')
        data = loadmat(data_file)

        images = data['affNISTdata']['image'].transpose().reshape(TOTAL_TRAINING_IMAGES, 40, 40, 1).astype(np.float32)
        labels = data['affNISTdata']['label_int'].astype(np.uint8)
        
        index_range = np.arange(len(images))
        idxs = np.random.choice(index_range, images_per_transformation)

        images_sample = images[idxs]
        labels_sample = labels[idxs]
        assert images_sample.shape == (images_per_transformation, 40, 40, 1)
        assert labels_sample.shape == (images_per_transformation,)
        images_sample = images_sample.reshape(images_per_transformation, 1600).transpose()
        assert images_sample.shape == (1600, images_per_transformation)

        peppered['affNISTdata']['image'] = np.append(peppered['affNISTdata']['image'], images_sample, axis=1)
        peppered['affNISTdata']['label_int'] = np.append(peppered['affNISTdata']['label_int'], labels_sample)

        num_img_peppered = t * images_per_transformation
        assert peppered['affNISTdata']['image'].shape == (1600, num_img_peppered + TOTAL_TRAINING_IMAGES)
        assert peppered['affNISTdata']['label_int'].shape == (num_img_peppered + TOTAL_TRAINING_IMAGES,)
        print('Generating... ' + str(int(num_img_peppered/num_img_to_pepper * 100)) + '%', end='\r')

    assert peppered['affNISTdata']['image'].shape == (1600, TOTAL_TRAINING_IMAGES + num_img_to_pepper)
    assert peppered['affNISTdata']['label_int'].shape == (TOTAL_TRAINING_IMAGES + num_img_to_pepper,)

    save_file = os.path.join(SAVE_DIR, str(percentage_of_transformed_images) + '_percent.mat')
    spio.savemat(save_file, peppered)


def check_output_dir(percentage_centered, percentage_transformed):
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    if percentage_centered != 100:
        filename = str(percentage_centered) + '_percent_centered_' + str(percentage_transformed) + '_percent_transformed.mat'
        peppered_file = os.path.join(SAVE_DIR, filename)
    else:
        filename = str(percentage_transformed) + '_percent.mat'
        peppered_file = os.path.join(SAVE_DIR, filename)
    
    if os.path.exists(peppered_file):
        print('Error: '+ peppered_file +' exists, remove manually to not overwrite')
        sys.exit()


if __name__ == '__main__':
    generate_peppered(100, 30)
