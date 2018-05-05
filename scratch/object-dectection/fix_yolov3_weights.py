#!/usr/bin/env python3

import h5py
import numpy as np

def scorrespond(conv_num, prefix, standalone_num, replace='darknet53'):

    b = '{}_standalone{}'.format(prefix, standalone_num)
    bb = '{}_standalone{}'.format(replace, standalone_num)

    correspondence = []

    oconv0 = '{}_conv{}_weight'.format(prefix, conv_num)
    obn_b0 = '{}_bn{}_bias'.format(prefix, conv_num)
    obn_m0 = '{}_bn{}_running_mean'.format(prefix, conv_num)
    obn_v0 = '{}_bn{}_running_var'.format(prefix, conv_num)
    obn_w0 = '{}_bn{}_weight'.format(prefix, conv_num)
    
    dconv0 = '{}_instance0_conv{}_weight'.format(bb, 0, 0)
    dbn_b0 = '{}_instance0_bn{}_bias'.format(bb, 0, 0)
    dbn_m0 = '{}_instance0_bn{}_running_mean'.format(bb, 0, 0)
    dbn_v0 = '{}_instance0_bn{}_running_var'.format(bb, 0, 0)
    dbn_w0 = '{}_instance0_bn{}_weight'.format(bb, 0, 0)
    
    correspondence += [ (oconv0, dconv0),
                        (obn_b0, dbn_b0),
                        (obn_m0, dbn_m0),
                        (obn_v0, dbn_v0),
                        (obn_w0, dbn_w0),
                       ]

    return correspondence

        
def bcorrespond(conv_range, prefix, block_num, block_range):

    b = '{}_block{}'.format(prefix, block_num)

    print(len(list(range(conv_range[0], conv_range[1], 2))))
    print(len(list(range(block_range[0], block_range[1]))))
    assert(len(list(range(conv_range[0], conv_range[1], 2))) == len(list(range(block_range[0], block_range[1]))))
    
    correspondence = []
    for conv_i, block_i in zip(range(conv_range[0], conv_range[1], 2), range(block_range[0], block_range[1])):

        oconv0 = '{}_conv{}_weight'.format(prefix, conv_i)
        obn_b0 = '{}_bn{}_bias'.format(prefix, conv_i)
        obn_m0 = '{}_bn{}_running_mean'.format(prefix, conv_i)
        obn_v0 = '{}_bn{}_running_var'.format(prefix, conv_i)
        obn_w0 = '{}_bn{}_weight'.format(prefix, conv_i)
        
        oconv1 = '{}_conv{}_weight'.format(prefix, conv_i+1)
        obn_b1 = '{}_bn{}_bias'.format(prefix, conv_i+1)
        obn_m1 = '{}_bn{}_running_mean'.format(prefix, conv_i+1)
        obn_v1 = '{}_bn{}_running_var'.format(prefix, conv_i+1)
        obn_w1 = '{}_bn{}_weight'.format(prefix, conv_i+1)
        
        dconv0 = '{}_instance{}_conv{}_weight'.format(b, block_i, 0)
        dbn_b0 = '{}_instance{}_bn{}_bias'.format(b, block_i, 0)
        dbn_m0 = '{}_instance{}_bn{}_running_mean'.format(b, block_i, 0)
        dbn_v0 = '{}_instance{}_bn{}_running_var'.format(b, block_i, 0)
        dbn_w0 = '{}_instance{}_bn{}_weight'.format(b, block_i, 0)
        
        dconv1 = '{}_instance{}_conv{}_weight'.format(b, block_i, 1)
        dbn_b1 = '{}_instance{}_bn{}_bias'.format(b, block_i, 1)
        dbn_m1 = '{}_instance{}_bn{}_running_mean'.format(b, block_i, 1)
        dbn_v1 = '{}_instance{}_bn{}_running_var'.format(b, block_i, 1)
        dbn_w1 = '{}_instance{}_bn{}_weight'.format(b, block_i, 1)
        
        correspondence += [ (oconv0, dconv0),
                            (obn_b0, dbn_b0),
                            (obn_m0, dbn_m0),
                            (obn_v0, dbn_v0),
                            (obn_w0, dbn_w0),
                            (oconv1, dconv1),
                            (obn_b1, dbn_b1),
                            (obn_m1, dbn_m1),
                            (obn_v1, dbn_v1),
                            (obn_w1, dbn_w1)
                          ]

    return correspondence

    
darknet_correspondence = []

darknet_correspondence += scorrespond(0, 'darknet53', 0)
darknet_correspondence += scorrespond(1,'darknet53', 1)
darknet_correspondence += bcorrespond((2, 4),'darknet53', 0, (0, 1))
darknet_correspondence += scorrespond(4,'darknet53', 2)
darknet_correspondence += bcorrespond((5, 9),'darknet53', 1, (0, 2))
darknet_correspondence += scorrespond(9,'darknet53', 3)
darknet_correspondence += bcorrespond((10, 26),'darknet53', 2, (0, 8))
darknet_correspondence += scorrespond(26,'darknet53', 4)
darknet_correspondence += bcorrespond((27, 43),'darknet53', 3, (0, 8))
darknet_correspondence += scorrespond(43,'darknet53', 5)
darknet_correspondence += bcorrespond((44, 52),'darknet53', 4, (0, 4))

darknet_correspondence += scorrespond(52, 'darknet53', 0, replace='yolo')
darknet_correspondence += scorrespond(53, 'yolo', 1, replace='yolo')
darknet_correspondence += scorrespond(54, 'yolo', 2, replace='yolo')
darknet_correspondence += scorrespond(55, 'yolo', 3, replace='yolo')
darknet_correspondence += scorrespond(56, 'yolo', 4, replace='yolo')
darknet_correspondence += scorrespond(57, 'yolo', 5, replace='yolo')
print('\n'.join(list(map(lambda x: str(x), darknet_correspondence))))
darknet_correspondence += [('yolo_conv58_bias', 'yolo_standalone6_instance0_conv0_bias'),
                           ('yolo_conv58_weight', 'yolo_standalone6_instance0_conv0_weight')]

with h5py.File('/home/jemmons/yolov3_raw.h5', 'r') as f: 
    with h5py.File('/home/jemmons/yolov3_new.h5', 'w') as new: 

        for c in darknet_correspondence:
            data = f[c[0]]

            new.create_dataset(c[1], data=data)
        
