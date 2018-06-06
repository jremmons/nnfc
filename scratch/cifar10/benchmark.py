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
import perflib
import sys
import glob

from PIL import Image

import cifar10_utils
from cifar10_networks import cifar10_networks as networks

import torch; 
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import numpy as np

logging.basicConfig(level=logging.DEBUG)
try:
    use_cuda = torch.cuda.is_available()
except:
    use_cuda = False

N = 10
    
def test(model, loss_fn, testloader):

    model.eval()

    compressed_sizes = []
    
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
        compressed_sizes += model.get_compressed_sizes()
        
        test_loss += loss_fn(output, batch_labels).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(batch_labels.data.view_as(pred)).long().cpu().sum().item()

    logging.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, count,
        100. * correct / count))

    t2 = timeit.default_timer()
    logging.info('Test Epoch took {} seconds'.format(t2-t1))

    print(np.median(np.asarray(compressed_sizes)))
    print(np.mean(np.asarray(compressed_sizes)))
    
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
        print(json.dumps(test_log, indent=4, sort_keys=True))
        
        ##############################################################
        # record the execution time
        ##############################################################
        torch.set_num_threads(1)
        input_x = np.random.randn(1,3,32,32).astype(np.float32)
        input_x = torch.autograd.Variable(torch.from_numpy(input_x)).cpu()

        net = net.cpu()
        out = net(input_x)
        times = []
        for _ in range(N):
            t1 = timeit.default_timer()
            out = net(input_x)
            t2 = timeit.default_timer()
            times.append(t2 - t1)

        times = np.asarray(times)
            
        print(json.dumps({
            'name' : 'time', 
            'count' : N,
            'mean' : np.mean(times),
            'median' : np.mean(times),
            'std' : np.std(times),
        }, indent=4, sort_keys=True))            
        
        # ##############################################################
        # # record the number of cycles
        # ##############################################################
        # input_x = np.random.randn(1,3,32,32).astype(np.float32)
        # input_x = torch.autograd.Variable(torch.from_numpy(input_x)).cpu()

        # instruction_counter = perflib.PerfCounter(counter_name='LIBPERF_COUNT_HW_INSTRUCTIONS')
        
        # net = net.cpu()
        # out = net(input_x)
        # instructions = []
        # for _ in range(N):
        #     instruction_counter.start()
        #     out = net(input_x)
        #     instruction_counter.stop()
        #     instructions.append(instruction_counter.getval())
        #     instruction_counter.reset()
            
        # instructions = np.asarray(instructions)
            
        # print(json.dumps({
        #     'name' : 'instructions', 
        #     'count' : N,
        #     'mean' : np.mean(instructions),
        #     'median' : np.mean(instructions),
        #     'std' : np.std(instructions),
        # }, indent=4, sort_keys=True))

        # ##############################################################
        # # record the number of instructions retired
        # ##############################################################
        # input_x = np.random.randn(1,3,32,32).astype(np.float32)
        # input_x = torch.autograd.Variable(torch.from_numpy(input_x)).cpu()

        # cycle_counter = perflib.PerfCounter(counter_name='LIBPERF_COUNT_HW_CPU_CYCLES')
        
        # net = net.cpu()
        # out = net(input_x)
        # cycles = []
        # for _ in range(N):
        #     cycle_counter.start()
        #     out = net(input_x)
        #     cycle_counter.stop()
        #     cycles.append(cycle_counter.getval())
        #     cycle_counter.reset()
            
        # cycles = np.asarray(cycles)
            
        # print(json.dumps({
        #     'name' : 'cycles', 
        #     'count' : N,
        #     'mean' : np.mean(cycles),
        #     'median' : np.mean(cycles),
        #     'std' : np.std(cycles),
        # }, indent=4, sort_keys=True))
        
        
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
