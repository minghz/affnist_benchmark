import argparse
import os
import sys
import zipfile
from six.moves import urllib

# affnist dataset
HOMEPAGE="https://www.cs.toronto.edu/~tijmen/affNIST/"
AFFNIST_CENTERED_TRAIN_URL = HOMEPAGE + "32x/just_centered/test.mat.zip"
AFFNIST_CENTERED_TEST_URL = HOMEPAGE + "32x/just_centered/training_and_validation.mat.zip"
AFFNIST_TRANSFORMED_TRAIN_URL = HOMEPAGE + "32x/transformed/training_and_validation_batches.zip"
AFFNIST_TRANSFORMED_TEST_URL = HOMEPAGE + "32x/transformed/test_batches.zip"


def download_and_uncompress_zip(URL, dataset_dir, force=False):
    '''
    Args:
        URL: the download links for data
        dataset_dir: the path to save data
        force: re-download data
    '''

    dirname = URL.split('/')[-2]
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    filename = URL.split('/')[-1]
    filepath = os.path.join(dataset_dir, dirname, filename)
    extract_to = os.path.splitext(filepath)[0]

    def download_progress(count, block_size, total_size):
        sys.stdout.write("\r>> Downloading %s %.1f%%" % (filename, float(count * block_size) / float(total_size) * 100.))
        sys.stdout.flush()

    if not force and os.path.exists(filepath):
        print("file %s already exist" % (filename))
    else:
        filepath, _ = urllib.request.urlretrieve(URL, filepath, download_progress)
        print()
        print('Successfully Downloaded', filename)

    with zipfile.ZipFile(filepath, 'r') as fd:
        print('Unzipping', filename)
        fd.extractall(dirname)
        fd.close()
        print('Successfully unzipped')
        print()

#    with gzip.open(filepath, 'rb') as f_in, open(extract_to, 'wb') as f_out:
#        print('Extracting ', filename)
#        shutil.copyfileobj(f_in, f_out)
#        print('Successfully extracted')
#        print()

def start_download(save_to, force):
    download_and_uncompress_zip(AFFNIST_CENTERED_TRAIN_URL, save_to, force)
    download_and_uncompress_zip(AFFNIST_CENTERED_TEST_URL, save_to, force)
    download_and_uncompress_zip(AFFNIST_TRANSFORMED_TRAIN_URL, save_to, force)
    download_and_uncompress_zip(AFFNIST_TRANSFORMED_TEST_URL, save_to, force)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script for automatically downloading datasets')
    parser.add_argument("--save_to", default='.')
    parser.add_argument("--force", default=False, type=bool)
    args = parser.parse_args()
    start_download(args.save_to, args.force)
