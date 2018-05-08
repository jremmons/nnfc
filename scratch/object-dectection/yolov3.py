import os
import h5py
import timeit
import itertools
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from PIL import Image, ImageDraw
from timeit import Timer

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

def dn_register_weights_helper(register_p, register_b, block_name, i, conv, bn):
    register_p('{}_conv{}_weight'.format(block_name, i), conv.weight)
    register_b('{}_bn{}_running_mean'.format(block_name, i), bn.running_mean)
    register_b('{}_bn{}_running_var'.format(block_name, i), bn.running_var)
    register_b('{}_bn{}_weight'.format(block_name, i), bn.weight)
    register_b('{}_bn{}_bias'.format(block_name, i), bn.bias)

class DarknetBlock(nn.Module):
    def __init__(self, block_name, nFilter1, nFilter2, activaction_func=nn.LeakyReLU(0.1)):
        super(DarknetBlock, self).__init__()

        self.block_name = block_name
        self.activaction_func = activaction_func

        self.conv = [
            nn.Conv2d(nFilter1, nFilter2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(nFilter2, nFilter1, kernel_size=3, stride=1, padding=1, bias=False)
        ]

        self.bn = [
            nn.BatchNorm2d(nFilter2),
            nn.BatchNorm2d(nFilter1)
        ]

    def register_weights(self, register_p, register_b):
        for i, (conv, bn) in enumerate(zip(self.conv, self.bn)):
            dn_register_weights_helper(register_p, register_b, self.block_name, i, conv, bn)

    def forward(self, x):
        out = self.activaction_func(self.bn[0](self.conv[0](x)))
        out = self.activaction_func(self.bn[1](self.conv[1](out)))
        out = out + x

        return out


class DarknetConv(nn.Module):
    def __init__(self, conv_name, nFilter1, nFilter2, stride=2, activaction_func=nn.LeakyReLU(0.1)):
        super(DarknetConv, self).__init__()

        self.conv_name = conv_name
        self.activaction_func = activaction_func

        self.conv = nn.Conv2d(nFilter1, nFilter2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(nFilter2)

    def register_weights(self, register_p, register_b):
        dn_register_weights_helper(register_p, register_b, self.conv_name, 0, self.conv, self.bn)

    def forward(self, x):

        out = self.activaction_func(self.bn(self.conv(x)))
        return out


class YoloBlock(nn.Module):
    def __init__(self, conv_name, nFilter1, nFilter2, size, stride, padding, activaction_func=nn.LeakyReLU(0.1)):
        super(YoloBlock, self).__init__()

        self.conv_name = conv_name
        self.activaction_func = activaction_func

        self.conv = nn.Conv2d(nFilter1, nFilter2, kernel_size=size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(nFilter2)

    def register_weights(self, register_p, register_b):
        dn_register_weights_helper(register_p, register_b, self.conv_name, 0, self.conv, self.bn)

    def forward(self, x):
        out = self.activaction_func(self.bn(self.conv(x)))
        return out


class YoloConv(nn.Module):
    def __init__(self, conv_name, nFilter1, nFilter2, activaction_func=lambda x: x):
        super(YoloConv, self).__init__()

        self.conv_name = conv_name
        self.activaction_func = activaction_func

        self.conv = nn.Conv2d(nFilter1, nFilter2, kernel_size=1, stride=1, padding=0, bias=True)

    def register_weights(self, register_p, register_b):
        register_p('{}_conv0_weight'.format(self.conv_name), self.conv.weight)
        register_p('{}_conv0_bias'.format(self.conv_name), self.conv.bias)

    def forward(self, x):
        out = self.activaction_func(self.conv(x))
        return out

class YoloUpsample(nn.Module):
    def __init__(self):
        super(YoloUpsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def register_weights(self, register_p, register_b):
        pass

    def forward(self, x):
        return self.upsample(x)

class YoloV3(nn.Module):
    def __init__(self):
        super(YoloV3, self).__init__()

        # darknet53 layers (the first 52 conv layers are present)
        self.dn53_standalone = [
            [DarknetConv('darknet53_standalone0_instance0', 3, 32, stride=1)],
            [DarknetConv('darknet53_standalone1_instance0', 32, 64)],
            [DarknetConv('darknet53_standalone2_instance0', 64, 128)],
            [DarknetConv('darknet53_standalone3_instance0', 128, 256)],
            [DarknetConv('darknet53_standalone4_instance0', 256, 512)],
            [DarknetConv('darknet53_standalone5_instance0', 512, 1024)]
        ]

        self.dn53_block = []

        # darknet53_block_structure = [1, 2, 8, 8, 4]
        for i, k in enumerate([1, 2, 8, 8, 4]):
            self.dn53_block += [[
                DarknetBlock('darknet53_block{}_instance{}'.format(i, j),
                    2 ** (i + 6), 2 ** (i + 5)) for j in range(k)
            ]]

        # YOLO extracts features from intermediate points in Darknet53
        # The blocks below are layer where the features are taken from
        self.layer36 = (self.dn53_standalone[0] +
                        self.dn53_standalone[1] +
                        self.dn53_block[0] +
                        self.dn53_standalone[2] +
                        self.dn53_block[1] +
                        self.dn53_standalone[3] +
                        self.dn53_block[2])

        self.layer61 = (self.dn53_standalone[4] +
                        self.dn53_block[3])

        self.layer74 = (self.dn53_standalone[5] +
                        self.dn53_block[4])

        # yolo detection layers
        # detection 1
        self.layer79 = [
            YoloBlock('yolo_standalone0_instance0', 1024, 512, 1, 1, 0),
            YoloBlock('yolo_standalone1_instance0', 512, 1024, 3, 1, 1),
            YoloBlock('yolo_standalone2_instance0', 1024, 512, 1, 1, 0),
            YoloBlock('yolo_standalone3_instance0', 512, 1024, 3, 1, 1),
            YoloBlock('yolo_standalone4_instance0', 1024, 512, 1, 1, 0)
        ]

        self.layer80 = [YoloBlock('yolo_standalone5_instance0', 512, 1024, 3, 1, 1)]
        self.layer81 = [YoloConv('yolo_standalone6_instance0', 1024, 255)]


        # detection 2
        self.layer85 = [
            YoloBlock('yolo_standalone7_instance0', 512, 256, 1, 1, 0),
            YoloUpsample()
        ]

        self.layer91 = [
            YoloBlock('yolo_standalone8_instance0', 768, 256, 1, 1, 0),
            YoloBlock('yolo_standalone9_instance0', 256, 512, 3, 1, 1),
            YoloBlock('yolo_standalone10_instance0', 512, 256, 1, 1, 0),
            YoloBlock('yolo_standalone11_instance0', 256, 512, 3, 1, 1),
            YoloBlock('yolo_standalone12_instance0', 512, 256, 1, 1, 0)
        ]

        self.layer92 = [YoloBlock('yolo_standalone13_instance0', 256, 512, 3, 1, 1)]
        self.layer93 = [YoloConv('yolo_standalone14_instance0', 512, 255)]

        # detection 3
        self.layer97 = [
            YoloBlock('yolo_standalone15_instance0', 256, 128, 1, 1, 0),
            YoloUpsample()
        ]

        self.layer104 = [
            YoloBlock('yolo_standalone16_instance0', 384, 128, 1, 1, 0),
            YoloBlock('yolo_standalone17_instance0', 128, 256, 3, 1, 1),
            YoloBlock('yolo_standalone18_instance0', 256, 128, 1, 1, 0),
            YoloBlock('yolo_standalone19_instance0', 128, 256, 3, 1, 1),
            YoloBlock('yolo_standalone20_instance0', 256, 128, 1, 1, 0),
            YoloBlock('yolo_standalone21_instance0', 128, 256, 3, 1, 1)
        ]

        self.layer105 = [YoloConv('yolo_standalone22_instance0', 256, 255)]

        #register the layers
        for layer in itertools.chain(self.layer36, self.layer61,
            self.layer74, self.layer79, self.layer80, self.layer81,
            self.layer85, self.layer91, self.layer92, self.layer93,
            self.layer97, self.layer104, self.layer105):
            layer.register_weights(self.register_parameter, self.register_buffer)


    @staticmethod
    def apply_layers(layers, x):
        for layer in layers:
            x = layer(x)

        return x


    @staticmethod
    def process_prediction(prediction, anchors):
        batch_size = prediction.size(0)

        inp_dim = 416
        num_classes = 80

        assert prediction.size(2) ==  prediction.size(3), 'height and width must be the same'
        stride = inp_dim // prediction.size(2)
        grid_size = inp_dim // stride

        bbox_attrs = 5 + num_classes
        num_anchors = len(anchors)

        anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

        prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
        prediction = prediction.transpose(1,2).contiguous()
        prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

        prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
        prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
        prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

        grid_len = np.arange(grid_size)
        a,b = np.meshgrid(grid_len, grid_len)

        x_offset = torch.FloatTensor(a).view(-1,1)
        y_offset = torch.FloatTensor(b).view(-1,1)

        x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

        prediction[:,:,:2] += x_y_offset

        anchors = torch.FloatTensor(anchors)
        anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
        prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

        prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))
        prediction[:,:,:4] *= stride

        return prediction


    def forward(self, x):
        # get the intermediates from the darknet featurizer
        layer36 = YoloV3.apply_layers(self.layer36, x)
        layer61 = YoloV3.apply_layers(self.layer61, layer36)
        layer74 = YoloV3.apply_layers(self.layer74, layer61)

        # process the intermediates for detections
        layer79 = YoloV3.apply_layers(self.layer79, layer74)
        predict0 = YoloV3.apply_layers(self.layer80 + self.layer81, layer79)

        layer85 = YoloV3.apply_layers(self.layer85, layer79)
        layer86 = torch.cat((layer85, layer61), 1)

        layer91 = YoloV3.apply_layers(self.layer91, layer86)
        predict1 = YoloV3.apply_layers(self.layer92 + self.layer93, layer91)

        layer97 = YoloV3.apply_layers(self.layer97, layer91)
        layer98 = torch.cat((layer97, layer36), 1)

        predict2 = YoloV3.apply_layers(self.layer104 + self.layer105, layer98)

        # process the detections
        detections0 = YoloV3.process_prediction(predict0, [(116,90), (156,198), (373,326)])
        detections1 = YoloV3.process_prediction(predict1, [(30,61), (62,45), (59,119)])
        detections2 = YoloV3.process_prediction(predict2, [(10,13), (16,30), (33,23)])

        return torch.cat((detections0, detections1, detections2), 1)

def load_model():
    yolov3 = YoloV3()

    # load the weights
    with h5py.File('yolov3.h5', 'r') as f:
        model_params = yolov3.state_dict()
        for param_name in model_params.keys():

            weights = torch.from_numpy(np.asarray(f[param_name]).astype(np.float32))
            model_params[param_name].data.copy_(weights)

    return yolov3

def dump_detections(y, img):
    for i in range(y.shape[1]):
        coords = y[0, i, :4]
        objectness = y[0, i, 4]
        classes = y[0, i, 5:]

        detections = []
        if objectness > 0.6:
            confidences = y[0, i, 5:].detach().numpy()
            confidences /= sum(confidences)
            idx = np.argmax(confidences)

            print('"' + coco_names[idx] + '"',
                  '(' + str(confidences[idx]) + ')',
                  y[0, i, :4].detach().numpy())

            x1 = y[0, i, 0] - y[0, i ,2] / 2
            x2 = y[0, i, 0] + y[0, i ,2] / 2
            y1 = y[0, i, 1] - y[0, i ,3] / 2
            y2 = y[0, i, 1] + y[0, i ,3] / 2

            draw = ImageDraw.Draw(img)
            draw.rectangle(((x1, y1), (x2, y2)))

def main():
    parser = argparse.ArgumentParser('Run YoloV3 on input image.')
    parser.add_argument('image', nargs='+')
    args = parser.parse_args()

    yolov3 = load_model()

    for img_path in args.image:
        img_original = Image.open(img_path).convert('RGB')
        img_original = img_original.resize((416, 416))
        img = np.asarray(img_original)
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img).float().div(255.0)
        x = Variable(img)

        print('>> %s:' % img_path)

        t1 = timeit.default_timer()
        y = yolov3(x)
        print('inference time:', timeit.default_timer() - t1)

        dump_detections(y, img_original)
        img_original.save(os.path.splitext(img_path)[0] + '.out.jpg')

if __name__ == '__main__':
    main()
