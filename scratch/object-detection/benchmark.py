#!/usr/bin/env python3

import os
import argparse

def main(options):
    pass

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('images')
    parser.add_argument('labels')
    parser.add_argument('--codec', type=str, default=None)
    parser.add_argument('--compression-layer-index', type=int, default=None)
    return parser.parse_args()
