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

def parse_labels(label, size):
    labels = []
    for line in label.split("\n")[:-1]:
        data = line.strip().split(' ')
        bb_o = [float(x) for x in data[1:]]
        idx = int(data[0])

        bb = [0] * 4
        bb[0] = size * (bb_o[0] - bb_o[2] / 2)
        bb[1] = size * (bb_o[1] - bb_o[3] / 2)
        bb[2] = size * (bb_o[0] + bb_o[2] / 2)
        bb[3] = size * (bb_o[1] + bb_o[3] / 2)

        labels += [{
            'coco_idx': idx,
            'bb': bb
        }]

    return labels

def main(images_path, labels_path):
    size = 416

    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data = DataLoader(YoloDataset(images_path, labels_path, t, size), batch_size=64, shuffle=True, num_workers=mp.cpu_count())
    model = yolo.load_model()
    model.to(device)

    count = [0] * 80
    correct = [0] * 80

    with torch.no_grad():
        for i, (local_batch, local_labels) in enumerate(data):

            local_batch = local_batch.to(device)
            output = model(local_batch)

            for j, detections in enumerate(utils.parse_detections(output)):
                detections = utils.non_max_suppression(detections, confidence_threshold=0.2)

                for target in parse_labels(local_labels[0][j], size):

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

            print('[%d/%d] mAP: %.6f (%.6f)' % (i + 1, len(data), psum / 80, sum(correct)/sum(count)))


if __name__ == '__main__':
    main(images_path=sys.argv[1], labels_path=sys.argv[2])
