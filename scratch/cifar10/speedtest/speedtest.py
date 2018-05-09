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

import torch; torch.set_num_threads(1)
import torch.nn.functional as F
# import torch.optim as optim
import sys

import resnet
import mobilenet
import simplenet

N = 3

def main(args):

    #resnet_blocks = [3,8,36,3]
    #resnet_blocks = [3,4,23,3]
    #resnet_blocks = [3,4,6,3]
    #resnet_blocks = [2,2,2,2]

    #input_x = np.random.randn(1,3,32,32).astype(np.float32)
    input_x = np.random.randn(1,3,416,416).astype(np.float32)
    input_x = torch.autograd.Variable(torch.from_numpy(input_x)).cpu()

    #net = simplenet.SimpleNet9()
    #net = resnet.ResNet18()
    #net = resnet.ResNet50()
    #net = resnet.ResNet101()
    #net = resnet.ResNet152()
    net = mobilenet.MobileNet()
    
    time = []
    for _ in range(N):
        t1 = timeit.default_timer()
        output = net(input_x)
        t2 = timeit.default_timer()
        time.append(t2-t1)
    time = np.asarray(time)
    sys.stdout.write(str(np.median(time)) + ',' + str(np.average(time)) + ',' + str(np.std(time))+'\n')

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

    


