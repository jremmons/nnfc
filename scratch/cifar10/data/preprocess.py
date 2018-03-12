import argparse 
import h5py
import os
import json
import six
import time

import numpy as np
from PIL import Image

TRAIN_FILENAMES = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
TEST_FILENAMES = ['test_batch']

NUM2LABEL = {
    0 : 'airplane',
    1 : 'automobile',
    2 : 'bird',
    3 : 'cat',
    4 : 'deer',
    5 : 'dog',
    6 : 'frog',
    7 : 'horse',
    8 : 'ship',
    9 : 'truck',
    }

    
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        return dict

def data_preprocess(filenames):

    labels = []
    raw_data = []
    for filename in filenames:

        data = unpickle(filename)
        
        labels += data[b'labels']
        raw_data.append(data[b'data'])

    count = 10000 * len(filenames) # cifar10 has 10k elements in each batch file

    raw_data = np.concatenate(raw_data, axis=0)    
    data_red = np.reshape(raw_data[:, :1024], (count, 32,32,1))
    data_green = np.reshape(raw_data[:, 1024:2048], (count, 32,32,1))
    data_blue = np.reshape(raw_data[:, 2048:], (count, 32,32,1))

    raw_data = np.concatenate([data_red, data_green, data_blue], axis=-1).astype(np.uint8)
    labels = np.asarray(labels).astype(np.uint32)

    return {
        'raw_data' : raw_data,
        'labels' : labels
        }

    
def main(args):

    train_filenames = list(map(lambda x: os.path.join(args.data_dir, x), TRAIN_FILENAMES))
    test_filenames = list(map(lambda x: os.path.join(args.data_dir, x), TEST_FILENAMES))


    train_data = data_preprocess(train_filenames)
    test_data = data_preprocess(test_filenames)

    d = {
        'train_data_raw' : train_data['raw_data'],
        'train_data_labels' : train_data['labels'],
        'test_data_raw' : test_data['raw_data'],
        'test_data_labels' : test_data['labels'],
        }

    a = {
        'num2label' : json.dumps(NUM2LABEL, indent=4, sort_keys=True),
        'description' : 'The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.',
        'url' : 'http://www.cs.toronto.edu/~kriz/cifar.html',
        'datetime_created' : time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        'preprocess_author' : 'John R. Emmons',
        'preprocess_email' : 'jemmons@cs.stanford.edu',
        'comments' : 
'''
The train and test arrays are RGB format.
Use the num2label dictionary to see the label_number to label_string correspondence.
The datetime_created is the time the preprocessing script was run in GMT time.'''
    }

    with h5py.File(args.output_hdf5, 'w') as f:

        # add data members
        for varname in d.keys():
            f.create_dataset(varname, data=d[varname])

        # add attributes (stored as datasets for ease of use on cli with h5utils)
        for varname in a.keys():
            f.create_dataset(varname, data=a[varname])

        f.create_dataset('hdf5_version', data=six.u(h5py.version.hdf5_version))
        f.create_dataset('h5py_version', data=six.u(h5py.version.version))
            
    print('done!')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('data_dir', type=str)
    parser.add_argument('output_hdf5', type=str)
    
    args = parser.parse_args()
    main(args)
