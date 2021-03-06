import argparse
import h5py
import os
import json
import shutil
import logging
import time
import timeit

# import perflib

import numpy as np

import torch; torch.set_num_threads(1)
import torch.nn.functional as F
import sys

import resnet
import resnet_with_compression
import mobilenet
import mobilenetv2
import simplenet

N = 100

def main(args):

    #resnet_blocks = [3,8,36,3]
    #resnet_blocks = [3,4,23,3]
    #resnet_blocks = [3,4,6,3]
    #resnet_blocks = [2,2,2,2]

    input_x = np.random.randn(1,3,32,32).astype(np.float32)
    #input_x = np.random.randn(1,3,416,416).astype(np.float32)
    input_x = torch.autograd.Variable(torch.from_numpy(input_x)).cpu()

    tests = [
        ('simplenet7_thin', simplenet.SimpleNet7_thin()),
        ('simplenet7', simplenet.SimpleNet7()),
        ('simplenet9_mobile', simplenet.SimpleNet9_mobile()),
        ('simplenet9_thin', simplenet.SimpleNet9_thin()),
        ('simplenet9', simplenet.SimpleNet9()),
        ('resnet18', resnet.ResNet18()),
        ('resnet18JPEG', resnet_with_compression.ResNet18()),
        ('mobilenetv2', mobilenetv2.MobileNetV2()),
        ('mobilenet', mobilenet.MobileNet()),
    ]


    for test in tests:
        net_name = test[0]
        net = test[1].eval()
        
        time = []
        for _ in range(N):
            t1 = timeit.default_timer()
            output = net(input_x)
            t2 = timeit.default_timer()
            time.append(t2-t1)
        time = np.asarray(time)
        sys.stdout.write(net_name + ',' + str(np.median(time)) + ',' + str(np.average(time)) + ',' + str(np.std(time))+'\n')

    return 
    
    # term = range(8)
    # term_names = ['input', 'layer1']
    # layer_num = 1
    # for x in resnet_blocks:
    #     layer_num += x * 2
    #     term_names.append('layer{}'.format(layer_num))

    # term_names.append('pool')
    # term_names.append('output')
    # print(term_names)
        
    # #net = resnet.ResNet(7, True, block=resnet.Bottleneck, num_blocks=resnet_blocks).cpu()
    # net = resnet.ResNet(7, True, block=resnet.BasicBlock, num_blocks=resnet_blocks).cpu()
    # net(input_x)

    # p1_ = perflib.PerfCounter(counter_name='LIBPERF_COUNT_HW_CPU_CYCLES')
    # p2_ = perflib.PerfCounter(counter_name='LIBPERF_COUNT_HW_INSTRUCTIONS')


    # for term_layer in term:
    #     net = resnet.ResNet(term_layer, False, block=resnet.Bottleneck, num_blocks=resnet_blocks).cpu()        
    #     time = []
    #     for _ in range(N):
    #         t1 = timeit.default_timer()
    #         output = net(input_x)
    #         t2 = timeit.default_timer()
    #         time.append(t2-t1)
    #     time = np.asarray(time)
    #     sys.stdout.write(term_names[term_layer] +','+str(np.median(time)) + ',' + str(np.average(time)) + ',' + str(np.std(time))+'\n')

    # for term_layer in term:
    #     net = resnet.ResNet(term_layer, False, block=resnet.Bottleneck, num_blocks=resnet_blocks).cpu()        
    #     p = perflib.PerfCounter(counter_name='LIBPERF_COUNT_HW_CPU_CYCLES')
    #     time = []
    #     for _ in range(N):
    #         p.start()
    #         output = net(input_x)
    #         p.stop()
    #         time.append(p.getval())
    #         p.reset()
    #     time = np.asarray(time)
    #     sys.stdout.write(term_names[term_layer] +','+str(np.median(time)) + ',' + str(np.average(time)) + ',' + str(np.std(time))+'\n')

    # for term_layer in term:
    #     net = resnet.ResNet(term_layer, False, block=resnet.Bottleneck, num_blocks=resnet_blocks).cpu()        
    #     p = perflib.PerfCounter(counter_name='LIBPERF_COUNT_HW_INSTRUCTIONS')
    #     time = []
    #     for _ in range(N):
    #         p.start()
    #         output = net(input_x)
    #         p.stop()
    #         time.append(p.getval())
    #         p.reset()
    #     time = np.asarray(time)
    #     sys.stdout.write(term_names[term_layer] +','+str(np.median(time)) + ',' + str(np.average(time)) + ',' + str(np.std(time))+'\n')
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    main(args)

    


