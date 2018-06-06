#!/usr/bin/env python3

import os
import sys
import pprint

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import utils
import yolov3 as yolo

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

class YoloDataset(Dataset):
    def __init__(self, images_path, labels_path, size=416):
        self.images = []
        self.labels = []
        self.images_path = images_path
        self.labels_path = labels_path
        self.size = size
        self.max_items = 50

        for img in sorted(os.listdir(images_path)):
            lbl = os.path.basename(img).split('.')[0] + '.txt'
            label_path = os.path.join(labels_path, lbl)

            if os.path.exists(label_path):
                self.images += [img]
                self.labels += [lbl]

        assert(len(self.images) == len(self.labels))
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.images_path, self.images[index])
        label_path = os.path.join(self.labels_path, self.labels[index])

        labels = []
        with open(label_path) as label_file:
            labels += [label_file.read()]

        image = utils.normalize_image(image_path, self.size)            
        return image[0,:,:,:], labels

def main(images_path, labels_path):
    size = 416

    data = DataLoader(YoloDataset(images_path, labels_path, size), batch_size=32, shuffle=False)
    model = yolo.load_model()
    model.to(device)

    correct, total_targets = 0, 0

    for i, (local_batch, local_labels) in enumerate(data):
        with torch.no_grad():
            local_batch = local_batch.to(device)
            output = model(local_batch)

            # XXX normally the above line should be:
            # output = model(local_batch)
            # (processing the whole batch at once), but for some
            # reason it doesn't work. so the current code only
            # works for batch_size=1, which is far from ideal.

        detections = utils.parse_detections(output)
        detections = utils.non_max_suppression(detections)

        for target in parse_labels(local_labels[0][0], size):
            total_targets += 1
            for det in [det for det in detections if det.coco_idx == target['coco_idx']]:
                if utils.iou(target['bb'], det.bb) >= 0.5:
                    correct += 1
                    break

        if total_targets:
            print('[%d/%d] mAP: %.6f' % (i + 1, len(data), correct / total_targets))


if __name__ == '__main__':
    main(images_path=sys.argv[1], labels_path=sys.argv[2])
