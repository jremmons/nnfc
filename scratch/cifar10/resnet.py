'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from nnfc.modules.nnfc import NnfcEncoder
from nnfc.modules.nnfc import NnfcDecoder
# from nnfc.modules.noop import NoopEncoder
# from nnfc.modules.noop import NoopDecoder

import timeit

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        #out = self.bn2(self.conv2(out))
        #out += self.shortcut(x)
        #out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Autoencoder(nn.Module):

    def __init__(self, num_planes_input, activation_compaction_factor):
        super(Autoencoder, self).__init__()

        num_planes_layer1 = int(num_planes_input // activation_compaction_factor)
        num_planes_layer2 = int(num_planes_layer1 // activation_compaction_factor)
        num_planes_layer3 = int(num_planes_layer2 // activation_compaction_factor)

        self.noop_encoder = NnfcEncoder()
        self.noop_decoder = NnfcDecoder()
        # self.noop_encoder = NoopEncoder()
        # self.noop_decoder = NoopDecoder()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(num_planes_input, num_planes_input, 1, stride=1, padding=0),
            nn.BatchNorm2d(num_planes_input),
            nn.ReLU(True),
            nn.Conv2d(num_planes_input, num_planes_layer1, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_planes_layer1),
            nn.ReLU(True),
            #
            nn.Conv2d(num_planes_layer1, num_planes_layer1, 1, stride=1, padding=0),
            nn.BatchNorm2d(num_planes_layer1),
            nn.ReLU(True),
            nn.Conv2d(num_planes_layer1, num_planes_layer2, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_planes_layer2),
            nn.ReLU(True),
            #
            nn.Conv2d(num_planes_layer2, num_planes_layer2, 1, stride=1, padding=0),
            nn.BatchNorm2d(num_planes_layer2),
            nn.ReLU(True),
            nn.Conv2d(num_planes_layer2, num_planes_layer3, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_planes_layer3),
            nn.ReLU(True),
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_planes_layer3, num_planes_layer3, 1, stride=1, padding=0),
            nn.BatchNorm2d(num_planes_layer3),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_planes_layer3, num_planes_layer2, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_planes_layer2),
            nn.ReLU(True),
            #
            nn.ConvTranspose2d(num_planes_layer2, num_planes_layer2, 1, stride=1, padding=0),
            nn.BatchNorm2d(num_planes_layer2),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_planes_layer2, num_planes_layer1, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_planes_layer1),
            nn.ReLU(True),
            #
            nn.ConvTranspose2d(num_planes_layer1, num_planes_layer1, 1, stride=1, padding=0),
            nn.BatchNorm2d(num_planes_layer1),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_planes_layer1, num_planes_input, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_planes_input),
            nn.ReLU(True),
            )
        
    def forward(self, x):
        x = self.encoder(x)

        # use_gpu = x.is_cuda
        # x = self.noop_encoder(x, input_on_gpu=use_gpu)
        # x = self.noop_decoder(x, put_output_on_gpu=use_gpu)
        
        x = self.decoder(x)
        return x

    
class AutoencoderResNet(nn.Module):
    '''
    Default configuration is a resnet18-like model with an autoencoder
    after the conv_3 layer (see table 1 in https://arxiv.org/pdf/1512.03385.pdf).
    '''
    
    def __init__(self, compaction_factor=1, block=BasicBlock, num_blocks=[2,2,2,2], num_classes=10):
        super(AutoencoderResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

        self.autoencoder = Autoencoder(128, compaction_factor)

        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        return x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.autoencoder(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    

class ResNet(nn.Module):
    def __init__(self, term_layer, print_size, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

        self.autoencoder = Autoencoder(128, 2)

        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.print_size = print_size
        self.size_func = lambda x : functools.reduce(lambda a,b : a*b, x)
        self.early_term = term_layer        
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        layers = 0

        if self.print_size:
            print('input', x.shape, self.size_func(x.shape))
        if self.early_term == layers:
            return None
        else:
            layers += 1
            
        out = F.relu(self.bn1(self.conv1(x)))
        if self.print_size:
            print('layer0', out.shape, self.size_func(out.shape))
        if self.early_term == layers:
            return None
        else:
            layers += 1

        out = self.layer1(out)
        if self.print_size:
            print('layer1', out.shape, self.size_func(out.shape))
        if self.early_term == layers:
            return None
        else:
            layers += 1

        out = self.layer2(out)
        if self.print_size:
            print('layer2', out.shape, self.size_func(out.shape))
        if self.early_term == layers:
            return None
        else:
            layers += 1

        out = self.layer3(out)
        if self.print_size:
            print('layer3', out.shape, self.size_func(out.shape))
        if self.early_term == layers:
            return None
        else:
            layers += 1

        out = self.layer4(out)
        if self.print_size:
            print('layer4', out.shape, self.size_func(out.shape))
        if self.early_term == layers:
            return None
        else:
            layers += 1

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if self.print_size:
            print('pool', out.shape, self.size_func(out.shape))
        if self.early_term == layers:
            return None
        else:
            layers += 1

        out = self.linear(out)
        if self.print_size:
            print('output', out.shape, self.size_func(out.shape))
        if self.early_term == layers:
            return None
        else:
            layers += 1

        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()
