#!/usr/bin/env python3

import os
import json
import argparse
import collections
import types

from pprint import pprint

import numpy as np

import yolov3 as yolo
from test import do_test

from nnfc.modules.nnfc import CompressionLayer

def create_params_dict(str_data):
    if str_data == None:
        return {}

    return {x[0]: int(x[1]) for x in [y.split("=") for y in str_data.split(",")]}

def main(options):
    compression_layer = None

    compressed_sizes = []
    compression_times = []

    if options.codec:
        encoder_name, decoder_name = ('%s_encoder' % options.codec, '%s_decoder' % options.codec)
        compression_layer = CompressionLayer(
            encoder_name=encoder_name,
            encoder_params_dict=create_params_dict(options.encoder_params),
            decoder_name=decoder_name,
            decoder_params_dict=create_params_dict(options.decoder_params))

    def progress_callback(model, batch_number, batch_count, map_score):
        if options.codec:
            compression_times.append(model.timelogger.points[0][1])
            compressed_sizes.extend([x[0] for x in model.get_compressed_sizes()])

        print('[%d/%d] mAP: %.6f' % (batch_number, batch_count, map_score))

    model = yolo.load_model(options.compression_layer_index, compression_layer, True)
    map_score = do_test(model, options.images, options.labels, options.batch_size,
                        progress_callback)

    data = collections.OrderedDict([
        ('codec', options.codec),
        ('compression_at', options.compression_layer_index),
        ('batch_size', options.batch_size),
        ('encoder_params', options.encoder_params),
        ('decoder_params', options.decoder_params),
        ('images', options.images),
        ('labels', options.labels),
        ('map_score', map_score),
        ('compressed_size', {
            ('mean', np.mean(compressed_sizes)),
            ('median', np.median(compressed_sizes)),
            ('stdev', np.std(compressed_sizes)),
        }),
        ('compression_time', {
            ('mean', np.mean(compression_times)),
            ('median', np.median(compression_times)),
            ('stdev', np.std(compression_times)),
        }),
    ])

    if options.output:
        with open(options.output, "w") as outfile:
            json.dump(data, outfile)
    else:
        pprint(data)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('images')
    parser.add_argument('labels')
    parser.add_argument('--codec', type=str, default=None)
    parser.add_argument('--compression-layer-index', type=int, default=0,
                        help='insert the compression layer before this block')
    parser.add_argument('--encoder-params', type=str, default=None, help='A=X,B=Y,C=Z')
    parser.add_argument('--decoder-params', type=str, default=None, help='A=X,B=Y,C=Z')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output', type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments())
