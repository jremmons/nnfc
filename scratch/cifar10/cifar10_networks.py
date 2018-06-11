import lenet
import simplenet
import resnet
import resnet_with_compression
import mobilenet
import mobilenetv2
import densenet
import dpn
import preact_resnet

cifar10_networks = {
    'lenet' : lenet.LeNet(),
    'simplenet9' : simplenet.SimpleNet9(),
    'simplenet9_thin' : simplenet.SimpleNet9_thin(),
    'simplenet9_mobile' : simplenet.SimpleNet9_mobile(),
    'simplenet7' : simplenet.SimpleNet7(),
    'simplenet7_thin' : simplenet.SimpleNet7_thin(),
    'resnet18NNFC1' : resnet_with_compression.ResNet18NNFC1(),
    'resnet18EH0' : resnet_with_compression.ResNet18EH(layer=0, quantizer=20),
    'resnet18EH1' : resnet_with_compression.ResNet18EH(layer=1, quantizer=6),
    'resnet18EH2' : resnet_with_compression.ResNet18EH(layer=2, quantizer=5),
    'resnet18EH3' : resnet_with_compression.ResNet18EH(layer=3, quantizer=3),
    'resnet18EH4' : resnet_with_compression.ResNet18EH(layer=4, quantizer=10),
    'resnet18JPEG90' : resnet_with_compression.ResNet18JPEG(quantizer=90),
    'resnet18JPEG87' : resnet_with_compression.ResNet18JPEG(quantizer=87),
    'resnet18' : resnet.ResNet18(),
    'resnet101' : resnet.ResNet101(),
    'mobilenetslimplus' : mobilenet.MobileNetSlimPlus(),
    'mobilenetslim' : mobilenet.MobileNetSlim(),
    'mobilenet' : mobilenet.MobileNet(),
    'mobilenetv2' : mobilenetv2.MobileNetV2(),
    'densenet121' : densenet.DenseNet121(),
    'dpn92' : dpn.DPN92(),
    'preact_resnet18' : preact_resnet.PreActResNet18(),
}


