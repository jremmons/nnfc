'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from nnfc.modules.nnfc import CompressionLayer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
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

class ResNetJPEG(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, quantizer=100):
        super(ResNetJPEG, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.timing = False
        self.jpeg_image_compression_layer = CompressionLayer(encoder_name='jpeg_image_encoder',
                                                        encoder_params_dict={'quantizer' : quantizer},
                                                        decoder_name='jpeg_image_decoder',
                                                        decoder_params_dict={})

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_compressed_sizes(self):
        return self.jpeg_image_compression_layer.get_compressed_sizes()

    def forward(self, x):
        out = x
        out = self.jpeg_image_compression_layer(out)
        if self.timing:
            return out

        out = F.relu(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNetEH(nn.Module):
    def __init__(self, block, num_blocks, layer, num_classes=10, quantizer=100):
        super(ResNetEH, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.timing = False
        self.layer = layer
        self.jpeg_image_compression_layer = CompressionLayer(encoder_name='jpeg_encoder',
                                                        encoder_params_dict={'quantizer' : quantizer},
                                                        decoder_name='jpeg_decoder',
                                                        decoder_params_dict={})

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_compressed_sizes(self):
        return self.jpeg_image_compression_layer.get_compressed_sizes()

    def forward(self, x):
        out = x

        out = F.relu(self.bn1(self.conv1(out)))
        if self.layer == 0:
            out = self.jpeg_image_compression_layer(out)
            if self.timing:
                return out

        out = self.layer1(out)
        if self.layer == 1:
            out = self.jpeg_image_compression_layer(out)
            if self.timing:
                return out

        out = self.layer2(out)
        if self.layer == 2:
            out = self.jpeg_image_compression_layer(out)
            if self.timing:
                return out

        out = self.layer3(out)
        if self.layer == 3:
            out = self.jpeg_image_compression_layer(out)
            if self.timing:
                return out

        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        if self.layer == 4:
            out = self.jpeg_image_compression_layer(out)
            if self.timing:
                return out

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNetMPEG(nn.Module):
    def __init__(self, block, num_blocks, layer, num_classes=10, quantizer=24, codec='h264'):
        super(ResNetMPEG, self).__init__()

        encoder_name = 'avc_encoder' if codec == 'h264' else 'heif_encoder'
        decoder_name = 'avc_decoder' if codec == 'h264' else 'heif_decoder'

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.timing = False
        self.layer = layer
        self.jpeg_image_compression_layer = CompressionLayer(encoder_name=encoder_name,
                                                             encoder_params_dict={'quantizer' : quantizer},
                                                             decoder_name=decoder_name,
                                                             decoder_params_dict={})

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_compressed_sizes(self):
        return self.jpeg_image_compression_layer.get_compressed_sizes()

    def forward(self, x):
        out = x

        out = F.relu(self.bn1(self.conv1(out)))
        if self.layer == 0:
            out = self.jpeg_image_compression_layer(out)
            if self.timing:
                return out

        out = self.layer1(out)
        if self.layer == 1:
            out = self.jpeg_image_compression_layer(out)
            if self.timing:
                return out

        out = self.layer2(out)
        if self.layer == 2:
            out = self.jpeg_image_compression_layer(out)
            if self.timing:
                return out

        out = self.layer3(out)
        if self.layer == 3:
            out = self.jpeg_image_compression_layer(out)
            if self.timing:
                return out

        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        if self.layer == 4:
            out = self.jpeg_image_compression_layer(out)
            if self.timing:
                return out

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNetNNFC1(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, quantizer=87):
        super(ResNetNNFC1, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.timing = False
        self.nnfc_compression_layer = CompressionLayer(encoder_name='nnfc1_encoder',
                                                       encoder_params_dict={'quantizer' : quantizer},
                                                       decoder_name='nnfc1_decoder',
                                                       decoder_params_dict={})

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_compressed_sizes(self):
        return self.nnfc_compression_layer.get_compressed_sizes()

    def forward(self, x):
        out = x

        out = F.relu(self.bn1(self.conv1(out)))
        if self.timing:
            return out

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.nnfc_compression_layer(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18EH(layer=1, quantizer=100):
    return ResNetEH(BasicBlock, [2,2,2,2], layer=layer, quantizer=quantizer)

def ResNet18AVC(layer=1, quantizer=24):
    return ResNetMPEG(BasicBlock, [2,2,2,2], layer=layer, quantizer=quantizer, codec='h264')

def ResNet18HEIF(layer=1, quantizer=24):
    return ResNetMPEG(BasicBlock, [2,2,2,2], layer=layer, quantizer=quantizer, codec='h265')

def ResNet18JPEG(quantizer=100):
    return ResNetJPEG(BasicBlock, [2,2,2,2], quantizer=quantizer)

def ResNet18NNFC1(quantizer=100):
    return ResNetNNFC1(BasicBlock, [2,2,2,2], quantizer=quantizer)


# def ResNet18():
#     return ResNet(BasicBlock, [2,2,2,2])

# def ResNet34():
#     return ResNet(BasicBlock, [3,4,6,3])

# def ResNet50():
#     return ResNet(Bottleneck, [3,4,6,3])

# def ResNet101():
#     return ResNet(Bottleneck, [3,4,23,3])

# def ResNet152():
#     return ResNet(Bottleneck, [3,8,36,3])

# def test():
#     net = ResNet18()
#     y = net(Variable(torch.randn(1,3,32,32)))
#     print(y.size())

# test()
