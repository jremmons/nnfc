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
    #'resnet18JPEG' : resnet_with_compression.ResNet18(),
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


