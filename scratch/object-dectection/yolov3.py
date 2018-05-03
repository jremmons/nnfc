import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class DarknetBlock(nn.Module):
    def __init__(self, block_name, nFilter1, nFilter2, activaction_func=nn.LeakyReLU(0.1)):
        super(DarknetBlock, self).__init__()

        self.block_name = block_name
        self.activaction_func = activaction_func
        
        self.conv0 = nn.Conv2d(nFilter1, nFilter2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(nFilter2)
        self.conv1 = nn.Conv2d(nFilter2, nFilter1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nFilter1)

    def register_weights(self, register_p, register_b):

        register_p('{}_conv{}_weight'.format(self.block_name, 0), self.conv0.weight)

        register_b('{}_bn{}_running_mean'.format(self.block_name, 0), self.bn0.running_mean)
        register_b('{}_bn{}_running_var'.format(self.block_name, 0), self.bn0.running_var)
        register_b('{}_bn{}_weight'.format(self.block_name, 0), self.bn0.weight)
        register_b('{}_bn{}_bias'.format(self.block_name, 0), self.bn0.bias)

        register_p('{}_conv{}_weight'.format(self.block_name, 1), self.conv1.weight)

        register_b('{}_bn{}_running_mean'.format(self.block_name, 1), self.bn1.running_mean)
        register_b('{}_bn{}_running_var'.format(self.block_name, 1), self.bn1.running_var)
        register_b('{}_bn{}_weight'.format(self.block_name, 1), self.bn1.weight)
        register_b('{}_bn{}_bias'.format(self.block_name, 1), self.bn1.bias)        
        
    def forward(self, x):

        out = self.activaction_func(self.bn0(self.conv0(x)))
        out = self.activaction_func(self.bn1(self.conv1(out)))

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

        register_p('{}_conv0_weight'.format(self.conv_name), self.conv.weight)
        
        register_b('{}_bn0_running_mean'.format(self.conv_name), self.bn.running_mean)
        register_b('{}_bn0_running_var'.format(self.conv_name), self.bn.running_var)
        register_b('{}_bn0_weight'.format(self.conv_name), self.bn.weight)
        register_b('{}_bn0_bias'.format(self.conv_name), self.bn.bias)

    def forward(self, x):

        out = self.activaction_func(self.bn(self.conv(x)))
        return out

    
class YoloBlock(nn.Module):
    def __init__(self, block_name, nFilter1, nFilter2, activaction_func=nn.LeakyReLU(0.1)):
        super(YoloBlock, self).__init__()

        self.block_name = block_name
        self.activaction_func = activaction_func
        
        self.conv0 = nn.Conv2d(nFilter1, nFilter2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(nFilter2)
        self.conv1 = nn.Conv2d(nFilter2, nFilter1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nFilter1)

    def register_weights(self, register_p, register_b):

        register_p('{}_conv{}_weight'.format(self.block_name, 0), self.conv0.weight)

        register_b('{}_bn{}_running_mean'.format(self.block_name, 0), self.bn0.running_mean)
        register_b('{}_bn{}_running_var'.format(self.block_name, 0), self.bn0.running_var)
        register_b('{}_bn{}_weight'.format(self.block_name, 0), self.bn0.weight)
        register_b('{}_bn{}_bias'.format(self.block_name, 0), self.bn0.bias)

        register_p('{}_conv{}_weight'.format(self.block_name, 1), self.conv1.weight)

        register_b('{}_bn{}_running_mean'.format(self.block_name, 1), self.bn1.running_mean)
        register_b('{}_bn{}_running_var'.format(self.block_name, 1), self.bn1.running_var)
        register_b('{}_bn{}_weight'.format(self.block_name, 1), self.bn1.weight)
        register_b('{}_bn{}_bias'.format(self.block_name, 1), self.bn1.bias)        
        
    def forward(self, x):

        out = self.activaction_func(self.bn0(self.conv0(x)))
        out = self.activaction_func(self.bn1(self.conv1(out)))

        return out
    
class YoloConv(nn.Module):
    def __init__(self, conv_name, nFilter1, nFilter2, activaction_func=lambda x: x):
        super(YoloConv, self).__init__()

        self.conv_name = conv_name
        self.activaction_func = activaction_func

        self.conv = nn.Conv2d(nFilter1, nFilter2, kernel_size=1, stride=1, padding=0, bias=True)

    def register_weights(self, register_p, register_b):

        register_p('{}_conv0_weight'.format(self.conv_name), self.conv.weight)
        register_p('{}_conv0_bias'.format(self.conv_name), self.conv.weight)

    def forward(self, x):

        out = self.activaction_func(self.conv(x))
        return out

    
class YoloV3(nn.Module):
    
    darknet53_block_structure = [1, 2, 8, 8, 4]
    yolo_block_structure = [3, 3, 3]
    
    def __init__(self):
        super(YoloV3, self).__init__()
        
        # darknet53 layers (the first 52 conv layers are present)
        self.darknet53_standalone0 = [ DarknetConv('darknet53_standalone0', 3, 32, stride=1) ]
        self.darknet53_standalone1 = [ DarknetConv('darknet53_standalone1', 32, 64) ]

        self.darknet53_block0 = [ DarknetBlock('darknet53_block0_instance{}'.format(i), 64, 32) for i in range(YoloV3.darknet53_block_structure[0]) ]

        self.darknet53_standalone2 = [ DarknetConv('darknet53_standalone2', 64, 128) ]

        self.darknet53_block1 = [ DarknetBlock('darknet53_block1_instance{}'.format(i), 128, 64) for i in range(YoloV3.darknet53_block_structure[1]) ]

        self.darknet53_standalone3 = [ DarknetConv('darknet53_standalone3', 128, 256) ]

        self.darknet53_block2 = [ DarknetBlock('darknet53_block2_instance{}'.format(i), 256, 128) for i in range(YoloV3.darknet53_block_structure[2]) ]

        self.darknet53_standalone4 = [ DarknetConv('darknet53_standalone4', 256, 512) ]

        self.darknet53_block3 = [ DarknetBlock('darknet53_block3_instance{}'.format(i), 512, 256) for i in range(YoloV3.darknet53_block_structure[3]) ]

        self.darknet53_standalone5 = [ DarknetConv('darknet53_standalone5', 512, 1024) ]

        self.darknet53_block4 = [ DarknetBlock('darknet53_block4_instance{}'.format(i), 1024, 512) for i in range(YoloV3.darknet53_block_structure[4]) ]

        self.route36 = self.darknet53_standalone0 + \
                       self.darknet53_standalone1 + \
                       self.darknet53_block0 + \
                       self.darknet53_standalone2 + \
                       self.darknet53_block1 + \
                       self.darknet53_standalone3 + \
                       self.darknet53_block2
        
        self.route61 = self.darknet53_standalone4 + \
                       self.darknet53_block3

        self.route74 = self.darknet53_standalone5 + \
                       self.darknet53_block4

        # yolo detection layers 
        self.yolo_block0 = [ YoloBlock('yolo_block0_instance{}'.format(i), 1024, 512) for i in range(YoloV3.yolo_block_structure[0]) ]
        self.yolo_standalone0 = [ YoloConv('yolo_standalone0', 1024, 255) ]
        
        # self.yolo_block1 = [ YoloBlock('yolo_block1_instance{}'.format(i), 512, 256) for i in range(YoloV3.yolo_block_structure[1]) ]
        # self.yolo_standalone1 = [ YoloConv('yolo_standalone1', 512, 255) ]

        # self.yolo_block2 = [ YoloBlock('yolo_block2_instance{}'.format(i), 256, 128) for i in range(YoloV3.yolo_block_structure[2]) ]
        # self.yolo_standalone2 = [ YoloConv('yolo_standalone2', 256, 255) ]
        
        # register all layers
        # todo...
        
        
    @staticmethod
    def apply_layers(layers, x):

        for layer in layers:
            x = layer(x)

        return x
            

    @staticmethod
    def process_prediction(prediction):

        batch_size = prediction.size(0)
        anchors = [(116, 90), (156, 198), (373, 326)]
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
        route36 = YoloV3.apply_layers(self.route36, x)
        route61 = YoloV3.apply_layers(self.route61, route36)
        route74 = YoloV3.apply_layers(self.route74, route61)

        # process the intermediates for detections
        predictions0 = YoloV3.apply_layers(self.yolo_block0 + self.yolo_standalone0, route74)

        # for now only process detections from this first layer
        # detect1 = YoloV3.apply_layers(self.yolo_block0 + self.yolo_standalone0, route74)
        # detect2 = YoloV3.apply_layers(self.yolo_block0 + self.yolo_standalone0, route74)
        predictions0 = YoloV3.process_prediction(predictions0)
        
        return predictions0

    
if __name__ == '__main__':
    yolov3 = YoloV3()

    model_params = yolov3.state_dict()
    for param_name in model_params.keys():
        print(param_name, model_params[param_name].shape)

    x = Variable(torch.randn(1, 3, 416, 416))
    y = yolov3(x)

    print(y[0,0,:])

    
