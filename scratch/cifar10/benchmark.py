#!/usr/bin/env python3

import argparse
import h5py
import os
import json
import shutil
import json
import logging
import time
import timeit
import sys
import glob

from PIL import Image

import cifar10_utils
from cifar10_networks import cifar10_networks as networks

import torch; torch.set_num_threads(1)
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import numpy as np


logging.basicConfig(level=logging.DEBUG)
try:
    use_cuda = torch.cuda.is_available()
except:
    use_cuda = False


def test(model, loss_fn, testloader):

    model.eval()

    test_loss = 0
    correct = 0
    count = testloader.batch_size * len(testloader)
    t1 = timeit.default_timer()
    for batch_idx, batch in enumerate(testloader):

        batch_data = torch.autograd.Variable(batch[0])
        batch_labels = torch.autograd.Variable(batch[1])

        if use_cuda:
            batch_data = batch_data.cuda()
            batch_labels = batch_labels.cuda()

        output = model(batch_data)

        test_loss += loss_fn(output, batch_labels).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(batch_labels.data.view_as(pred)).long().cpu().sum().item()

    logging.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, count,
        100. * correct / count))

    t2 = timeit.default_timer()
    logging.info('Test Epoch took {} seconds'.format(t2-t1))

    return {
        'validation_top1' : correct / count,
        'validation_loss' : test_loss
        }


def main(checkpoint_dir, config):

    logging.info('loading data into memory')
    with h5py.File(config['data_hdf5'], 'r') as f:
        test_data_raw = np.asarray(f['test_data_raw'])
        test_data_labels = np.asarray(f['test_data_labels'])
    logging.info('done! (loading data into memory)')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = cifar10_utils.Cifar10(test_data_raw, test_data_labels, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    shutil.copy(config['data_hdf5'], checkpoint_dir)

    loss_fn = torch.nn.CrossEntropyLoss()

    initial_epoch = 0
    net = None

    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint-epoch*.h5'))
    checkpoints = list(reversed(sorted(checkpoints)))

    latest_checkpoint = checkpoints[0]
    logging.info('loading from last checkpoint: {}'.format(latest_checkpoint))

    pytorch_checkpoints = glob.glob(os.path.join(checkpoint_dir, 'pytorch_checkpoint-epoch*.pt'))
    pytorch_checkpoints = list(reversed(sorted(pytorch_checkpoints)))

    latest_pytorch_checkpoint = pytorch_checkpoints[0]
    logging.info('loading from last pytorch_checkpoint: {}'.format(latest_pytorch_checkpoint))

    latest_epoch = latest_checkpoint.split('.h5')[0].split('checkpoint-epoch')[-1]
    logging.info('checkpoint epoch: {}'.format(latest_epoch))
    latest_epoch = int(latest_epoch)

    latest_pytorch_epoch = latest_pytorch_checkpoint.split('.pt')[0].split('pytorch_checkpoint-epoch')[-1]
    logging.info('pytorch checkpoint epoch: {}'.format(latest_pytorch_epoch))
    latest_pytorch_epoch = int(latest_pytorch_epoch)

    # initialize the epoch to one after the last checkpoint
    assert latest_epoch == latest_pytorch_epoch
    initial_epoch = latest_epoch + 1

    # restore the parameters to the value was stored in hdf5 file
    checkpoint_filename = os.path.abspath(os.path.join(checkpoint_dir, latest_checkpoint))
    logging.info('restoring parameters: ' +  checkpoint_filename)

    net = networks[config['network_name']]
    if use_cuda:
        net = net.cuda()
        
    with h5py.File(checkpoint_filename, 'r') as f:

        model_params = net.state_dict()
        for key in net.state_dict().keys():
            model_params[key].data.copy_(torch.from_numpy(np.asarray(f[key])))


        ##############################################################
        # compute the accuracy on the test set
        ##############################################################
        test_log = test(net, loss_fn, testloader)

        ##############################################################
        # record the execution time
        ##############################################################
        # todo add

        ##############################################################
        # record the number of cycles
        ##############################################################
        # todo add


        ##############################################################
        # record the number of instructions retired
        ##############################################################
        # todo add
        
        
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('checkpoint_dir', type=str)
    args = parser.parse_args()

    metadata_filename = os.path.join(args.checkpoint_dir, 'metadata.json')
    logging.info('loading parameters from checkpoint: {}.'.format(metadata_filename))

    with open(metadata_filename, 'r') as f:
        config = json.loads(f.read())

    cifar10_utils.check_for_required_params(config)
    keys = list(config.keys())
    assert 'creation_time' in keys
    assert 'data_hdf5' in keys
    assert 'checkpoint_dir' in keys

    experiment_configuration = {}
    for key in config.keys():
        experiment_configuration[key] = config[key]
            
    logging.info('using the following configuration...')
    logging.info(json.dumps(experiment_configuration, indent=4, sort_keys=True))

    main(args.checkpoint_dir, experiment_configuration)
