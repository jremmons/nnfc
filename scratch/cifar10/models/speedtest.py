import argparse
import h5py
import os
import json
import shutil
import logging
import time
import timeit

import perflib

import numpy as np

import resnet
import torch; torch.set_num_threads(1)
import torch.nn.functional as F
import sys

import lenet
import vgg
import resnet
import googlenet
import mobilenet
import mobilenetv2
import densenet

def main(args):

    input_x = np.random.randn(1,3,32,32).astype(np.float32)
    input_x = torch.autograd.Variable(torch.from_numpy(input_x)).cpu()

    networks = [
        {
            'network_name' : 'lenet',
            'reported_accuracy' : '.7627',
            'net' : lenet.LeNet()
        },
        {
            'network_name' : 'vgg19',
            'reported_accuracy' : '.9353',
            'net' : vgg.VGG('VGG19')
        },
        {
            'network_name' : 'resnet18',
            'reported_accuracy' : '.9302',
            'net' : resnet.ResNet18()
        },
        {
            'network_name' : 'resnet50',
            'reported_accuracy' : '.9362',
            'net' : resnet.ResNet50()
        },
        {
            'network_name' : 'resnet101',
            'reported_accuracy' : '.9410',
            'net' : resnet.ResNet101()
        },
        {
            'network_name' : 'mobilenetv2',
            'reported_accuracy' : '.9443',
            'net' : mobilenetv2.MobileNetV2()
        },
        {
            'network_name' : 'densenet121',
            'reported_accuracy' : '.9504',
            'net' : densenet.DenseNet121()
        },
    ]


    p1_ = perflib.PerfCounter(counter_name='LIBPERF_COUNT_HW_CPU_CYCLES')
    p2_ = perflib.PerfCounter(counter_name='LIBPERF_COUNT_HW_INSTRUCTIONS')

    N = 100
    for network in networks:

        net = network['net']

        time = []
        for _ in range(N):
            t1 = timeit.default_timer()
            output = net(input_x)
            t2 = timeit.default_timer()
            time.append(t2-t1)
        time = np.asarray(time)
        sys.stdout.write(network['network_name'] + ',' + network['reported_accuracy'] + ',' + str(np.median(time)) + ',' + str(np.average(time)) + ',' + str(np.std(time))+'\n')

        p = perflib.PerfCounter(counter_name='LIBPERF_COUNT_HW_CPU_CYCLES')
        time = []
        for _ in range(N):
            p.start()
            output = net(input_x)
            p.stop()
            time.append(p.getval())
            p.reset()
        time = np.asarray(time)
        sys.stdout.write(network['network_name'] + ',' + network['reported_accuracy'] + ',' + str(np.median(time)) + ',' + str(np.average(time)) + ',' + str(np.std(time))+'\n')

        p = perflib.PerfCounter(counter_name='LIBPERF_COUNT_HW_INSTRUCTIONS')
        time = []
        for _ in range(N):
            p.start()
            output = net(input_x)
            p.stop()
            time.append(p.getval())
            p.reset()
        time = np.asarray(time)
        sys.stdout.write(network['network_name'] + ',' + network['reported_accuracy'] + ',' + str(np.median(time)) + ',' + str(np.average(time)) + ',' + str(np.std(time))+'\n')
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    
    args = parser.parse_args()
    main(args)

    


