import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from nnfc.modules.nnfc import CompressionLayer

class AutoEncoderI(nn.Module):

    def __init__(self):
        super(AutoEncoderI, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
        #self.w1 = nn.Parameter(torch.randn(1))
        #self.register_parameter('weight', self.w1)
        
    def forward(self, x):
        out = x
        #out = self.w1 + out
        out = self.conv1(out)

        return out

class AutoEncoder0(nn.Module):

    def __init__(self):
        super(AutoEncoder0, self).__init__()
        stride = 1

        # encoder
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_bn1 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_bn3 = nn.BatchNorm2d(64)


        # decoder
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.deconv_bn1 = nn.BatchNorm2d(64)

        self.deconv3 = nn.ConvTranspose2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.deconv_bn3 = nn.BatchNorm2d(256)

        
    def forward(self, x):
        out = x
        out = F.relu(self.conv_bn1(self.conv1(out)))
        out = F.relu(self.conv_bn3(self.conv3(out)))

        out = F.relu(self.deconv_bn1(self.deconv1(out)))
        out = F.relu(self.deconv_bn3(self.deconv3(out)))
        
        return out

class AutoEncoder1(nn.Module):

    def __init__(self):
        super(AutoEncoder1, self).__init__()
        stride = 1

        self.compression_layer = CompressionLayer(encoder_name='nnfc1_encoder',
                                                  encoder_params_dict={'quantizer' : -1},
                                                  decoder_name='nnfc1_decoder',
                                                  decoder_params_dict={})
        
        # encoder
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_bn1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv_bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv_bn4 = nn.BatchNorm2d(64)

        # decoder
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.deconv_bn1 = nn.BatchNorm2d(64)

        self.deconv2 = nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.deconv_bn2 = nn.BatchNorm2d(128)

        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.deconv_bn3 = nn.BatchNorm2d(128)

        self.deconv4 = nn.ConvTranspose2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.deconv_bn4 = nn.BatchNorm2d(256)

        
    def forward(self, x):
        out = x
        out = F.relu(self.conv_bn1(self.conv1(out)))
        out = F.relu(self.conv_bn2(self.conv2(out)))
        out = F.relu(self.conv_bn3(self.conv3(out)))
        out = F.relu(self.conv_bn4(self.conv4(out)))

        out = self.compression_layer(out)
        
        out = F.relu(self.deconv_bn1(self.deconv1(out)))
        out = F.relu(self.deconv_bn2(self.deconv2(out)))
        out = F.relu(self.deconv_bn3(self.deconv3(out)))
        out = F.relu(self.deconv_bn4(self.deconv4(out)))
        
        return out
