#!/usr/bin/env python3

import os
import sys
import timeit
import pprint
import argparse

import numpy as np
import multiprocessing as mp

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

from PIL import Image

import utils
import yolov3 as yolo

from utils import device
from dataset import YoloDataset

def main(options):
    image_size = 416

    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data = DataLoader(YoloDataset(options.images, options.labels, t, image_size),
                      batch_size=options.batch_size, shuffle=True,
                      num_workers=mp.cpu_count())

    model = yolo.load_model(load_weights=False)
    model.train()
    model.to(device)

    # values from:
    # [yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9,
                                dampening=0, weight_decay=0.0005)

    for epoch in range(options.epochs):
        for i, (local_batch, local_labels) in enumerate(data):
            optimizer.zero_grad()
            local_batch = local_batch.to(device)

            outputs = model(local_batch)
            # detections = yolo.YoloV3.get_detections(outputs)
            # print(detections.shape)

            loss = yolo.YoloV3.get_loss(outputs, local_labels)
            print(loss[0].shape)
            # print(loss.shape)
            # loss.backward()
            # optimizer.step()

            
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('images', type=str)
    parser.add_argument('labels', type=str)
    options = parser.parse_args()
    return options

if __name__ == '__main__':
    main(parse_arguments())
