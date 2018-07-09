#!/usr/bin/env python3

import torch
import numpy as np
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

coco_names = [
    'person',        'bicycle',       'car',           'motorbike',
    'aeroplane',     'bus',           'train',         'truck',
    'boat',          'traffic light', 'fire hydrant',  'stop sign',
    'parking meter', 'bench',         'bird',          'cat',
    'dog',           'horse',         'sheep',         'cow',
    'elephant',      'bear',          'zebra',         'giraffe',
    'backpack',      'umbrella',      'handbag',       'tie',
    'suitcase',      'frisbee',       'skis',          'snowboard',
    'sports ball',   'kite',          'baseball bat',  'baseball glove',
    'skateboard',    'surfboard',     'tennis racket', 'bottle',
    'wine glass',    'cup',           'fork',          'knife',
    'spoon',         'bowl',          'banana',        'apple',
    'sandwich',      'orange',        'broccoli',      'carrot',
    'hot dog',       'pizza',         'donut',         'cake',
    'chair',         'sofa',          'pottedplant',   'bed',
    'diningtable',   'toilet',        'tvmonitor',     'laptop',
    'mouse',         'remote',        'keyboard',      'cell phone',
    'microwave',     'oven',          'toaster',       'sink',
    'refrigerator',  'book',          'clock',         'vase',
    'scissors',      'teddy bear',    'hair drier',    'toothbrush',
]

def create_params_dict(str_data):
    if str_data == None:
        return {}

    return {x[0]: int(x[1]) for x in [y.split("=") for y in str_data.split(",")]}

class Box:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        assert(self.x >= 0)
        assert(self.y >= 0)
        assert(self.w >= 0)
        assert(self.h >= 0)

    def to_x1y1x2y2(self):
        return [self.x - self.w / 2,
                self.y - self.h / 2,
                self.x + self.w / 2,
                self.y + self.h / 2]

    def __str__(self):
        return "box[center=(%.5f, %.5f), dims=(%.5f, %.5f)]" % (self.x,
            self.y, self.w, self.h)

    def __repr__(self):
        return str(self)

class DetectionOutput:
    def __init__(self):
        self.coco_idx = None
        self.coco_name = None
        self.confidence = None
        self.bb = None

    def __str__(self):
        return '"' + self.coco_name + '" (' + \
               str(self.confidence) + ') ' + str(self.bb)

    def __repr__(self):
        return str(self)

def parse_detections(y):
    object_threshold = 0.2
    y = y.detach().cpu().numpy()

    all_detections = []
    for k in range(y.shape[0]):
        detections = []
        detection_indices = np.argwhere(y[k, :, 4] > object_threshold)

        for i in detection_indices:
            confidences = y[k, i, 5:][0]
            confidences /= sum(confidences)
            idx = np.argmax(confidences)

            det = DetectionOutput()
            det.coco_idx = idx
            det.coco_name = coco_names[idx]
            det.confidence = confidences[idx]
            det.bb = Box(*(y[k, i, :4][0]))
            detections += [det]

        all_detections += [detections]

    return all_detections

# box = [x, y, w, h]
def iou(box_1, box_2):
    f_area = lambda b: ((b[3] - b[1] + 1) * (b[2] - b[0] + 1))

    box_1 = box_1.to_x1y1x2y2()
    box_2 = box_2.to_x1y1x2y2()

    # intersection coordinates
    box_i = [max(box_1[0], box_2[0]), max(box_1[1], box_2[1]),
             min(box_1[2], box_2[2]), min(box_1[3], box_2[3])]

    return f_area(box_i) / (f_area(box_1) + f_area(box_2) - f_area(box_i))

def non_max_suppression(detections, confidence_threshold=0.25, iou_threshold=0.5):
    # throw out the results with confidence less than the threshold
    outputs = [det for det in detections if det.confidence >= confidence_threshold]

    if not outputs:
        return outputs

    final_outputs = []
    for label in {det.coco_idx for det in outputs}:
        label_outs = [det for det in outputs if det.coco_idx == label]
        label_outs.sort(key=lambda x: x.confidence, reverse=True)

        label_finals = []
        while label_outs:
            label_finals += [label_outs[0]]
            del label_outs[0]
            label_outs = [det for det in label_outs
                          if iou(label_finals[-1].bb, det.bb) < iou_threshold]

        final_outputs += label_finals

    return final_outputs

def parse_labels(label, size):
    labels = []
    for line in label.split("\n")[:-1]:
        data = line.strip().split(' ')
        bb_o = [float(x) * size for x in data[1:]]
        idx = int(data[0])

        labels += [{
            'coco_idx': idx,
            'bb': Box(*bb_o)
        }]

    return labels
