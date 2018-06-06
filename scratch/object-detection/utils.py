#!/usr/bin/env python3

import torch
import numpy as np
from PIL import Image

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

    object_threshold = 0.25
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

            bb = y[k, i, :4][0]
            det.bb = [
                bb[0] - bb[2] / 2, #x1
                bb[1] - bb[3] / 2, #y1
                bb[0] + bb[2] / 2, #x2
                bb[1] + bb[3] / 2  #y2
            ]

            detections += [det]

        all_detections += [detections]

    return all_detections

# box = [x1, y1, x2, y2]
def iou(box_1, box_2):
    f_area = lambda b: ((b[3] - b[1]) * (b[2] - b[0]))

    # intersection coordinates
    box_i = [max(box_1[0], box_2[0]), max(box_1[1], box_2[1]),
             min(box_1[2], box_2[2]), min(box_1[3], box_2[3])]

    return f_area(box_i) / (f_area(box_1) + f_area(box_2) - f_area(box_i))

# def normalize_image(img_path, size=416):

#     # TODO(jremmons) add image mean substraction to this function.
#     # TODO(jremmons) add any other data augmentation that yolo might do. 

#     img_original = Image.open(img_path).convert('RGB')
#     img_original = img_original.resize((size, size))
#     img = np.asarray(img_original)
#     img = img.transpose((2, 0, 1))
    
#     img = np.expand_dims(img, 0)
#     img = torch.from_numpy(img).float().div(255.0)
#     return img

def non_max_suppression(detections, confidence_threshold=0.25, iou_threshold=0.4):
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
