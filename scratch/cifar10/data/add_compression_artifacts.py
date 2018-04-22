import argparse 
import h5py
import os
import json
import shutil
import subprocess
import time


import multiprocessing as mp
import numpy as np
from PIL import Image


TMPFILE_PREFIX = 'proprocess'
TMPDIR = '/dev/shm'

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

    
def add_compression_artifacts(args):

    image, encoder_cmdline, decoder_cmdline = args

    pid = str(os.getpid())

    r = image[0,:,:]
    g = image[1,:,:]
    b = image[2,:,:]
    image = np.stack((r,g,b), axis=-1)
    image = Image.fromarray(image)
    
    tmp_save_path = os.path.join(TMPDIR, TMPFILE_PREFIX+'_'+pid)
    tmp_bmp1_save_path = tmp_save_path + '.1.bmp'
    tmp_bmp2_save_path = tmp_save_path + '.2.bmp'
    tmp_jpg_save_path = tmp_save_path + '.jpg'
    image.save(tmp_bmp1_save_path)

    encoder_cmdline = encoder_cmdline.copy()
    encoder_cmdline += ['-outfile', tmp_jpg_save_path, tmp_bmp1_save_path]

    decoder_cmdline = decoder_cmdline.copy()
    decoder_cmdline += ['-outfile', tmp_bmp2_save_path, tmp_jpg_save_path]

    subprocess.check_call(encoder_cmdline)
    compressed_size = os.path.getsize(tmp_jpg_save_path)
    subprocess.check_call(decoder_cmdline)

    preprocessed_image = Image.open(tmp_bmp2_save_path)

    os.remove(tmp_bmp1_save_path)
    os.remove(tmp_bmp2_save_path)
    os.remove(tmp_jpg_save_path)

    preprocessed_image = np.asarray(preprocessed_image)
    r = preprocessed_image[:,:,0]
    g = preprocessed_image[:,:,1]
    b = preprocessed_image[:,:,2]

    preprocessed_image = np.stack((r,g,b), axis=0)
    
    return {
        'raw_image' : preprocessed_image,
        'compressed_size' : compressed_size
        }
    
def main(args):

    a = {
        'num2label' : json.dumps(NUM2LABEL, indent=4, sort_keys=True),
        'description' : 'The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.',
        'url' : 'http://www.cs.toronto.edu/~kriz/cifar.html',
        'datetime_created' : time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        'preprocess_author' : 'John R. Emmons',
        'preprocess_email' : 'jemmons@cs.stanford.edu',
        'cmdline_encoder' : args.encoder_cmdline,
        'cmdline_decoder' : args.decoder_cmdline,
        'comments' : 
'''
The each image in `train_data_raw` and `test_data_raw` was:
    1 read from original_data/{`train_data_raw` and `test_data_raw`}
    2. compressed using cmdline_encoder
    3. decompressed using cmdline_decoder
    4. stored into `train_data_raw` and `test_data_raw`
The train and test arrays are RGB format.
Use the num2label dictionary to see the label_number to label_string correspondence.
The datetime_created is the time the preprocessing script was run in GMT time.
'''
    }
    

    with h5py.File(args.output_hdf5, 'w') as f:
        with h5py.File(args.original_hdf5, 'r') as f_original:

            # copy the contents of the original file into a group             
            original = f.create_group('original_data')
            for data in f_original.keys():
                original.create_dataset(str(data), data=f_original[data])
                
            train_data = np.asarray(f_original['train_data_raw'])
            train_labels = np.asarray(f_original['train_data_labels'])
            train_len = len(train_labels)
                
            test_data = np.asarray(f_original['test_data_raw'])
            test_labels = np.asarray(f_original['test_data_labels'])
            test_len = len(test_labels)
                
        # add metadata to the hdf5 file
        for varname in a.keys():
            f.create_dataset(varname, data=a[varname])

        # apply compression artifacts to each data elements (train and test)
        with mp.Pool() as pool:
            encoder_cmdline_split = args.encoder_cmdline.split()
            decoder_cmdline_split = args.decoder_cmdline.split()

            # train
            train_jobs = [ (train_data[i,:,:,:], encoder_cmdline_split, decoder_cmdline_split) for i in range(train_len) ]
            preprocessed_train = pool.map(add_compression_artifacts, train_jobs)

            train_images = np.stack(list(map(lambda x: x['raw_image'], preprocessed_train)))
            train_compressed_size = np.asarray(list(map(lambda x: x['compressed_size'], preprocessed_train)))

            # test
            test_jobs = [ (test_data[i,:,:,:], encoder_cmdline_split, decoder_cmdline_split) for i in range(test_len) ]
            preprocessed_test = pool.map(add_compression_artifacts, test_jobs)

            test_images = np.stack(list(map(lambda x: x['raw_image'], preprocessed_test)))
            test_compressed_size = np.asarray(list(map(lambda x: x['compressed_size'], preprocessed_test)))
                

        f.create_dataset('train_data_labels', data=train_labels)
        f.create_dataset('train_data_raw', data=train_images)
        f.create_dataset('train_data_compressed_size', data=train_compressed_size)
        f.create_dataset('train_data_compressed_size_mean', data=np.mean(train_compressed_size))
        f.create_dataset('train_data_compressed_size_std', data=np.std(train_compressed_size))

        f.create_dataset('test_data_labels', data=test_labels)
        f.create_dataset('test_data_raw', data=test_images) 
        f.create_dataset('test_data_compressed_size', data=test_compressed_size)
        f.create_dataset('test_data_compressed_size_mean', data=np.mean(test_compressed_size))
        f.create_dataset('test_data_compressed_size_std', data=np.std(test_compressed_size))
               
        print('Done!')
                  
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('encoder_cmdline', type=str)
    parser.add_argument('decoder_cmdline', type=str)
    parser.add_argument('original_hdf5', type=str)
    parser.add_argument('output_hdf5', type=str)
    
    args = parser.parse_args()
    main(args)
