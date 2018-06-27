#!/usr/bin/env python3

import os
import argparse

from pprint import pprint

import yolov3 as yolo
from test import do_test

from nnfc.modules.nnfc import CompressionLayer

def create_params_dict(str_data):
    if str_data == None:
        return {}

    return {x[0]: int(x[1]) for x in [y.split("=") for y in str_data.split(",")]}

def progress_callback(model, batch_number, batch_count, map_score):
    pprint(model.timelogger.points)
    pprint(model.get_compressed_sizes())

def main(options):
    compression_layer = None

    if options.codec:
        encoder_name, decoder_name = ('%s_encoder' % options.codec, '%s_decoder' % options.codec)
        compression_layer = CompressionLayer(
            encoder_name=encoder_name,
            encoder_params_dict=create_params_dict(options.encoder_params),
            decoder_name=decoder_name,
            decoder_params_dict=create_params_dict(options.decoder_params))

    model = yolo.load_model(options.compression_layer_index, compression_layer, True)
    do_test(model, options.images, options.labels, options.batch_size,
            progress_callback)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('images')
    parser.add_argument('labels')
    parser.add_argument('--codec', type=str, default=None)
    parser.add_argument('--compression-layer-index', type=int, default=0,
                        help='insert the compression layer before this block')
    parser.add_argument('--encoder-params', type=str, default=None, help='A=X,B=Y,C=Z')
    parser.add_argument('--decoder-params', type=str, default=None, help='A=X,B=Y,C=Z')
    parser.add_argument('--batch-size', type=int, default=32)
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments())
