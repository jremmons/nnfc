import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

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

        out = out + x         
        return out


class YoloConv(nn.Module):
    def __init__(self, conv_name, nFilter1, nFilter2, stride=2, activaction_func=nn.LeakyReLU(0.1)):
        super(YoloConv, self).__init__()

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
        

class Darknet53FeatureExtractor(nn.Module):
    def __init__(self, block_structure=[1, 2, 8, 8, 4]):
        super(Darknet53FeatureExtractor, self).__init__()

        self.standalone0 = [ YoloConv('darknet53_standalone0', 3, 32, stride=1) ]
        self.standalone1 = [ YoloConv('darknet53_standalone1', 32, 64) ]

        self.block0 = [ YoloBlock('darknet53_block0_instance{}'.format(i), 64, 32) for i in range(block_structure[0]) ]

        self.standalone2 = [ YoloConv('darknet53_standalone2', 64, 128) ]

        self.block1 = [ YoloBlock('darknet53_block1_instance{}'.format(i), 128, 64) for i in range(block_structure[1]) ]

        self.standalone3 = [ YoloConv('darknet53_standalone3', 128, 256) ]

        self.block2 = [ YoloBlock('darknet53_block2_instance{}'.format(i), 256, 128) for i in range(block_structure[2]) ]

        self.standalone4 = [ YoloConv('darknet53_standalone4', 256, 512) ]

        self.block3 = [ YoloBlock('darknet53_block3_instance{}'.format(i), 512, 256) for i in range(block_structure[3]) ]

        self.standalone5 = [ YoloConv('darknet53_standalone5', 512, 1024) ]

        self.block4 = [ YoloBlock('darknet53_block4_instance{}'.format(i), 1024, 512) for i in range(block_structure[4]) ]

        self.layers = self.standalone0 + \
                      self.standalone1 + \
                      self.block0 + \
                      self.standalone2 + \
                      self.block1 + \
                      self.standalone3 + \
                      self.block2 + \
                      self.standalone4 + \
                      self.block3 + \
                      self.standalone5 + \
                      self.block4
        
        for b in self.layers:
            b.register_weights(self.register_parameter, self.register_buffer)
        

    def register_weights(self, register_p, register_b):
        
        for b in self.layers:
            b.register_weights(register_p, register_b)
        
            
    def forward(self, x):

        out = x
        for layer in self.layers:
            out = layer(out)

        return out


class YoloV3(nn.Module):
    def __init__(self):
        super(YoloV3, self).__init__()

        # darknet53 layers (the first 52 conv layers are present)
        self.standalone0 = [ YoloConv('darknet53_standalone0', 3, 32, stride=1) ]
        self.standalone1 = [ YoloConv('darknet53_standalone1', 32, 64) ]

        self.block0 = [ YoloBlock('darknet53_block0_instance{}'.format(i), 64, 32) for i in range(block_structure[0]) ]

        self.standalone2 = [ YoloConv('darknet53_standalone2', 64, 128) ]

        self.block1 = [ YoloBlock('darknet53_block1_instance{}'.format(i), 128, 64) for i in range(block_structure[1]) ]

        self.standalone3 = [ YoloConv('darknet53_standalone3', 128, 256) ]

        self.block2 = [ YoloBlock('darknet53_block2_instance{}'.format(i), 256, 128) for i in range(block_structure[2]) ]

        self.standalone4 = [ YoloConv('darknet53_standalone4', 256, 512) ]

        self.block3 = [ YoloBlock('darknet53_block3_instance{}'.format(i), 512, 256) for i in range(block_structure[3]) ]

        self.standalone5 = [ YoloConv('darknet53_standalone5', 512, 1024) ]

        self.block4 = [ YoloBlock('darknet53_block4_instance{}'.format(i), 1024, 512) for i in range(block_structure[4]) ]

        self.route36 = self.standalone0 + \
                       self.standalone1 + \
                       self.block0 + \
                       self.standalone2 + \
                       self.block1 + \
                       self.standalone3 + \
                       self.block2
        
        self.route61 = self.standalone4 + \
                       self.block3

        self.route74 = self.standalone5 + \
                       self.block4

        # yolo detection layers 
        
        # register the layers here
        
        
    def forward(self, x):

        # get the intermediates from the darknet featurizer
        route36 = self.route36(x)
        route61 = self.route61(route36)
        route74 = self.self.route74(route61)

        # process the intermediate for detections


        
        return x

    
if __name__ == '__main__':
    yolov3 = YoloV3()

    model_params = yolov3.state_dict()
    for param_name in model_params.keys():
        print(param_name, model_params[param_name].shape)

    x = Variable(torch.randn(1, 3, 416, 416))
    y = yolov3(x)

    print(y.shape)

    
