import os
import sys
import time
import pprint
import itertools
import argparse

import h5py
import numpy as np
import torch; torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

import utils
from utils import device

class TimeLog:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.start = None
        self.points = []

    def begin(self):
        self.start = time.time()
        self.points = []

    def add_point(self, title):
        if not self.enabled:
            return

        assert self.start, "the timer has not been started"

        now = time.time()
        self.points += [(title, now - self.start)]
        self.start = now

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
            nn.Conv2d(nFilter1, nFilter2, kernel_size=1, stride=1, padding=0, bias=False).to(device),
            nn.Conv2d(nFilter2, nFilter1, kernel_size=3, stride=1, padding=1, bias=False).to(device)
        ]

        self.bn = [
            nn.BatchNorm2d(nFilter2).to(device),
            nn.BatchNorm2d(nFilter1).to(device)
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
    def __init__(self, conv_num, nFilter1, nFilter2, stride=2, activaction_func=nn.LeakyReLU(0.1)):
        super(DarknetConv, self).__init__()

        self.conv_name = 'darknet53_standalone%d_instance0' % conv_num
        self.activaction_func = activaction_func

        self.conv = nn.Conv2d(nFilter1, nFilter2, kernel_size=3, stride=stride, padding=1, bias=False).to(device)
        self.bn = nn.BatchNorm2d(nFilter2).to(device)

    def register_weights(self, register_p, register_b):
        dn_register_weights_helper(register_p, register_b, self.conv_name, 0, self.conv, self.bn)

    def forward(self, x):
        out = self.activaction_func(self.bn(self.conv(x)))
        return out

class YoloBlock(nn.Module):
    def __init__(self, conv_num, nFilter1, nFilter2, size, stride, padding, activaction_func=nn.LeakyReLU(0.1)):
        super(YoloBlock, self).__init__()

        self.conv_name = "yolo_standalone%d_instance0" % conv_num
        self.activaction_func = activaction_func

        self.conv = nn.Conv2d(nFilter1, nFilter2, kernel_size=size, stride=stride, padding=padding, bias=False).to(device)
        self.bn = nn.BatchNorm2d(nFilter2).to(device)

    def register_weights(self, register_p, register_b):
        dn_register_weights_helper(register_p, register_b, self.conv_name, 0, self.conv, self.bn)

    def forward(self, x):
        out = self.activaction_func(self.bn(self.conv(x)))
        return out

class YoloConv(nn.Module):
    def __init__(self, conv_num, nFilter1, nFilter2, activaction_func=lambda x: x):
        super(YoloConv, self).__init__()

        self.conv_name = "yolo_standalone%d_instance0" % conv_num
        self.activaction_func = activaction_func

        self.conv = nn.Conv2d(nFilter1, nFilter2, kernel_size=1, stride=1, padding=0, bias=True).to(device)

    def register_weights(self, register_p, register_b):
        register_p('{}_conv0_weight'.format(self.conv_name), self.conv.weight)
        register_p('{}_conv0_bias'.format(self.conv_name), self.conv.bias)

    def forward(self, x):
        out = self.activaction_func(self.conv(x))
        return out

class YoloUpsample(nn.Module):
    def __init__(self):
        super(YoloUpsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest').to(device)

    def register_weights(self, register_p, register_b):
        pass

    def forward(self, x):
        return self.upsample(x)

class YoloV3(nn.Module):
    anchors0 = torch.Tensor([(116,90), (156,198), (373,326)]).to(device)
    anchors1 = torch.Tensor([(30,61), (62,45), (59,119)]).to(device)
    anchors2 = torch.Tensor([(10,13), (16,30), (33,23)]).to(device)

    # If compression_layer_index != None, compression layer will be inserted
    # *before* that block.
    def __init__(self, load_weights=False, compression_layer_index=None,
                 compression_layer=None, log_time=False):
        super(YoloV3, self).__init__()

        if compression_layer and compression_layer_index:
            self.compression_layer_index = compression_layer_index
            self.compression_layer_class_name = compression_layer.__class__.__name__
        else:
            self.compression_layer_index = None
            self.compression_layer_class_name = None

        self.timelogger = TimeLog(log_time)

        # darknet53 layers (the first 52 conv layers are present)
        self.dn53_standalone = [
            [DarknetConv(0, 3, 32, stride=1)],
            [DarknetConv(1, 32, 64)],
            [DarknetConv(2, 64, 128)],
            [DarknetConv(3, 128, 256)],
            [DarknetConv(4, 256, 512)],
            [DarknetConv(5, 512, 1024)]
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
        self.layers = [None] * 13

        self.layers[0] = (
            self.dn53_standalone[0] +
            self.dn53_standalone[1] +
            self.dn53_block[0] +
            self.dn53_standalone[2] +
            self.dn53_block[1] +
            self.dn53_standalone[3] +
            self.dn53_block[2]
        )

        self.layers[1] = (
            self.dn53_standalone[4] +
            self.dn53_block[3]
        )

        self.layers[2] = (
            self.dn53_standalone[5] +
            self.dn53_block[4]
        )

        # yolo detection layers
        # detection 1
        self.layers[3] = [
            YoloBlock(0, 1024, 512, 1, 1, 0),
            YoloBlock(1, 512, 1024, 3, 1, 1),
            YoloBlock(2, 1024, 512, 1, 1, 0),
            YoloBlock(3, 512, 1024, 3, 1, 1),
            YoloBlock(4, 1024, 512, 1, 1, 0)
        ]

        self.layers[4] = [YoloBlock(5, 512, 1024, 3, 1, 1)]
        self.layers[5] = [YoloConv(6, 1024, 255)]

        # detection 2
        self.layers[6] = [
            YoloBlock(7, 512, 256, 1, 1, 0),
            YoloUpsample()
        ]

        self.layers[7] = [
            YoloBlock(8, 768, 256, 1, 1, 0),
            YoloBlock(9, 256, 512, 3, 1, 1),
            YoloBlock(10, 512, 256, 1, 1, 0),
            YoloBlock(11, 256, 512, 3, 1, 1),
            YoloBlock(12, 512, 256, 1, 1, 0)
        ]

        self.layers[8] = [YoloBlock(13, 256, 512, 3, 1, 1)]
        self.layers[9] = [YoloConv(14, 512, 255)]

        # detection 3
        self.layers[10] = [
            YoloBlock(15, 256, 128, 1, 1, 0),
            YoloUpsample()
        ]

        self.layers[11] = [
            YoloBlock(16, 384, 128, 1, 1, 0),
            YoloBlock(17, 128, 256, 3, 1, 1),
            YoloBlock(18, 256, 128, 1, 1, 0),
            YoloBlock(19, 128, 256, 3, 1, 1),
            YoloBlock(20, 256, 128, 1, 1, 0),
            YoloBlock(21, 128, 256, 3, 1, 1)
        ]

        self.layers[12] = [YoloConv(22, 256, 255)]

        #register the layers
        for layers in self.layers:
            for layer in layers:
                layer.register_weights(self.register_parameter, self.register_buffer)

        # adding the compression layer
        if self.compression_layer_index != None:
            self.compression_layer = compression_layer

            i = 0
            for layers in self.layers:
                j = 0
                for layer in layers:
                    if i == compression_layer_index:
                        layers.insert(j, self.compression_layer)
                        return

                    i += 1
                    j += 1

    def apply_layers(self, layers, x):
        for layer in layers:
            x = layer(x)

            if (self.compression_layer_index and
                layer.__class__.__name__ == self.compression_layer_class_name):
                self.timelogger.add_point('compression_done')

        return x

    
    @staticmethod
    def get_loss(predictions, targets):
        return predictions

    
    @staticmethod
    def get_detections(predictions):

        detections0 = YoloV3.process_prediction(predictions[0], YoloV3.anchors0)
        detections1 = YoloV3.process_prediction(predictions[1], YoloV3.anchors1)
        detections2 = YoloV3.process_prediction(predictions[2], YoloV3.anchors2)

        out = torch.cat((detections0, detections1, detections2), 1)
        return out

    
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

        prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
        prediction = prediction.transpose(1,2).contiguous()
        prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

        prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
        prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
        prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

        grid_len = np.arange(grid_size)
        a,b = np.meshgrid(grid_len, grid_len)

        x_offset = torch.FloatTensor(a).view(-1,1)
        y_offset = torch.FloatTensor(b).view(-1,1)

        x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

        prediction[:,:,:2] += torch.FloatTensor(x_y_offset).to(device)

        anchors = torch.FloatTensor(anchors).to(device)
        anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
        prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

        prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))
        prediction[:,:,:4] *= stride

        return prediction

    
    def forward(self, x):
        self.timelogger.begin()

        # DARKNET-53
        layer0 = self.apply_layers(self.layers[0], x)
        layer1 = self.apply_layers(self.layers[1], layer0)
        layer2 = self.apply_layers(self.layers[2], layer1)

        # YoloV3
        # process the intermediates for detections
        layer3 = self.apply_layers(self.layers[3], layer2)
        predict0 = self.apply_layers(self.layers[4] + self.layers[5], layer3)

        layer6 = self.apply_layers(self.layers[6], layer3)
        layer86 = torch.cat((layer6, layer1), 1)

        layer7 = self.apply_layers(self.layers[7], layer86)
        predict1 = self.apply_layers(self.layers[8] + self.layers[9], layer7)

        layer10 = self.apply_layers(self.layers[10], layer7)
        layer12 = torch.cat((layer10, layer0), 1)

        predict2 = self.apply_layers(self.layers[11] + self.layers[12], layer12)

        self.timelogger.add_point('network_done')
        return (predict0, predict1, predict2)

    
    def timelog(self):
        return self.timelog

  
    def get_compressed_sizes(self):
        return self.compression_layer.get_compressed_sizes()

    
def load_model(*args, **kwargs):
    yolov3 = YoloV3(*args, **kwargs)

    with h5py.File('yolov3.h5', 'r') as f:
        model_params = yolov3.state_dict()
        # print(list(model_params.keys()))
        # print(len(list(model_params.keys())))
        
        for param_name in model_params.keys():
            weights = torch.from_numpy(np.asarray(f[param_name]).astype(np.float32))
            model_params[param_name].data.copy_(weights)

    return yolov3

def main():
    parser = argparse.ArgumentParser('Run YoloV3 on input image.')
    parser.add_argument('image', nargs='+')
    args = parser.parse_args()

    print("* Running on:", device)

    import timeit
    
    yolov3 = load_model()
    for img_path in args.image:
        x = Variable(utils.normalize_image(img_path)).to(device)
        t1 = timeit.default_timer()
        y = yolov3(x)
        t2 = timeit.default_timer()
        print(t2-t1)
        detections = utils.parse_detections(y)[0]
        detections = utils.non_max_suppression(detections)
        pprint.pprint(detections)
        #img_original.save(os.path.splitext(img_path)[0] + '.out.jpg')

if __name__ == '__main__':
    main()
