#!/usr/bin/env python3

import os
import sys
import pprint

import numpy as np
import multiprocessing as mp

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

from PIL import Image

import utils
import timeit
import yolov3 as yolo

from utils import device
from dataset import YoloDataset

def do_test(model, images_path, labels_path, progress_callback=None):
    size = 416

    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data = DataLoader(YoloDataset(images_path, labels_path, t, size), batch_size=64, shuffle=True, num_workers=mp.cpu_count())

    count = [0] * 80
    correct = [0] * 80

    with torch.no_grad():
        for i, (local_batch, local_labels) in enumerate(data):

            local_batch = local_batch.to(device)
            output = model(local_batch)

            for j, detections in enumerate(utils.parse_detections(output)):
                detections = utils.non_max_suppression(detections, confidence_threshold=0.2)

                for target in utils.parse_labels(local_labels[0][j], size):
                    count[target['coco_idx']] += 1
                    for det in [det for det in detections if det.coco_idx == target['coco_idx']]:
                        if utils.iou(target['bb'], det.bb) >= 0.5:
                            correct[target['coco_idx']] += 1
                            break

            psum = 0
            for j in range(80):
                if count[j] == 0:
                    psum = -80
                    break

                p = correct[j] / count[j]
                psum += p

            if progress_callback:
                progress_callback(model, batch_number=(i + 1), batch_count=len(data), map_score=psum / 80)

    return psum / 80

def progress_callback(model, batch_number, batch_count, map_score):
    print('[%d/%d] mAP: %.6f' % (batch_number, batch_count, map_score))

def main(images_path, labels_path):
    model = yolo.load_model()
    model.to(device)

    do_test(model, images_path, labels_path, progress_callback)
    return 0

if __name__ == '__main__':
    sys.exit(main(images_path=sys.argv[1], labels_path=sys.argv[2]))
