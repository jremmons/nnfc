#!/usr/bin/env python3

import argparse
import h5py
import os
import json
import shutil
import json
import logging
import time
import timeit
import sys
import glob

from PIL import Image

import torch
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import numpy as np

import simplenet
import resnet
import mobilenetv2
import densenet
import dpn
import preact_resnet

logging.basicConfig(level=logging.DEBUG)
use_cuda = torch.cuda.is_available()

# TODO(jremmons) add a better programmatic interface for defining network architecture

networks = {
    'simplenet9' : simplenet.SimpleNet9(),
    'resnet18' : resnet.ResNet18(),
    'resnet101' : resnet.ResNet101(),
    'mobilenetv2' : mobilenetv2.MobileNetV2(),
    'densenet121' : densenet.DenseNet121(),
    'densenet121' : densenet.DenseNet121(),
    'dpn92' : dpn.DPN92(),
    'preact_resnet18' : preact_resnet.PreActResNet18(),
    }
        

class Cifar10(torch.utils.data.Dataset):

    def __init__(self, data_raw, data_labels, transform=None):

        r = data_raw[:,0,:,:]
        g = data_raw[:,1,:,:]
        b = data_raw[:,2,:,:]
        
        self.data_raw = np.stack([r,g,b], axis=-1)
        self.data_labels = data_labels
        self.transform = transform

    def __len__(self):

        return len(self.data_raw)
    
    def __getitem__(self, idx):

        image = Image.fromarray(self.data_raw[idx,:,:,:])
        
        if self.transform:
            image = self.transform(image)

        return image, self.data_labels[idx].astype(np.int64)
    

def train(model, epoch, loss_fn, optimizer, trainloader):

    model.train()

    train_loss = 0
    correct = 0
    count = trainloader.batch_size * len(trainloader)
    t1 = timeit.default_timer()
    for batch_idx, batch in enumerate(trainloader):

        batch_data = torch.autograd.Variable(batch[0])
        batch_labels = torch.autograd.Variable(batch[1])

        if use_cuda:
            batch_data = batch_data.cuda()
            batch_labels = batch_labels.cuda()

        optimizer.zero_grad()
        output = model(batch_data)
        loss = loss_fn(output, batch_labels)

        train_loss += loss.data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(batch_labels.data.view_as(pred)).long().sum().item()

        loss.backward()
        optimizer.step()

        if batch_idx % 5 == 0:
            logging.info('Train Epoch: {} [(lr: {}) {}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, optimizer.defaults['lr'], trainloader.batch_size * batch_idx, count,
                100. * trainloader.batch_size * batch_idx / count, loss.data.item()))

    t2 = timeit.default_timer()
    logging.info('Train Epoch took {} seconds'.format(t2-t1))

    return {
        'train_top1' : correct / count,
        'train_loss' : train_loss
        }
    
    
def test(model, loss_fn, testloader):

    model.eval()
    
    test_loss = 0
    correct = 0
    count = testloader.batch_size * len(testloader)
    t1 = timeit.default_timer()
    for batch_idx, batch in enumerate(testloader):

        batch_data = torch.autograd.Variable(batch[0])
        batch_labels = torch.autograd.Variable(batch[1])
        
        if use_cuda:
            batch_data = batch_data.cuda()
            batch_labels = batch_labels.cuda()        

        t1 = timeit.default_timer()
        output = model(batch_data)
        t2 = timeit.default_timer()
        #print('fwd:', t2-t1)

        test_loss += loss_fn(output, batch_labels).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(batch_labels.data.view_as(pred)).long().cpu().sum().item()
        
    logging.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, count,
        100. * correct / count))

    t2 = timeit.default_timer()
    logging.info('Test Epoch took {} seconds'.format(t2-t1))

    return {
        'validation_top1' : correct / count,
        'validation_loss' : test_loss
        }

    
def main(checkpoint_dir, resume, config):

    logging.info('loading data into memory')
    with h5py.File(config['data_hdf5'], 'r') as f:
        train_data_raw = np.asarray(f['train_data_raw'])
        train_data_labels = np.asarray(f['train_data_labels'])
        test_data_raw = np.asarray(f['test_data_raw'])
        test_data_labels = np.asarray(f['test_data_labels'])
    logging.info('done! (loading data into memory)')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = Cifar10(train_data_raw, train_data_labels, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=2)

    testset = Cifar10(test_data_raw, test_data_labels, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    
    shutil.copy(config['data_hdf5'], checkpoint_dir)
    
    #optimizer = optim.Adam(net.parameters(), lr=args.lr)
    #optimizer = optim.Adagrad(net.parameters(), lr=args.lr, lr_decay=0.01)
    #optimizer = optim.RMSprop(net.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    initial_epoch = 0
    net = None

    if not resume:
        net = networks[config['network_name']]
        if use_cuda:
            net.cuda()

    elif resume:
        checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint-epoch*.h5'))
        checkpoints = list(reversed(sorted(checkpoints)))

        latest_checkpoint = checkpoints[0]
        logging.info('loading from last checkpoint: {}'.format(latest_checkpoint))

        pytorch_checkpoints = glob.glob(os.path.join(checkpoint_dir, 'pytorch_checkpoint-epoch*.pt'))
        pytorch_checkpoints = list(reversed(sorted(pytorch_checkpoints)))

        latest_pytorch_checkpoint = pytorch_checkpoints[0]
        logging.info('loading from last pytorch_checkpoint: {}'.format(latest_pytorch_checkpoint))

        latest_epoch = latest_checkpoint.split('.h5')[0].split('checkpoint-epoch')[-1]
        logging.info('checkpoint epoch: {}'.format(latest_epoch))
        latest_epoch = int(latest_epoch)

        latest_pytorch_epoch = latest_pytorch_checkpoint.split('.pt')[0].split('pytorch_checkpoint-epoch')[-1]
        logging.info('pytorch checkpoint epoch: {}'.format(latest_pytorch_epoch))
        latest_pytorch_epoch = int(latest_pytorch_epoch)

        # initialize the epoch to one after the last checkpoint
        assert latest_epoch == latest_pytorch_epoch
        initial_epoch = latest_epoch + 1

        # restore the parameters to the value was stored in hdf5 file
        checkpoint_filename = os.path.abspath(os.path.join(checkpoint_dir, latest_checkpoint))
        logging.info('restoring parameters')
        net = torch.load(latest_pytorch_checkpoint)
        with h5py.File(checkpoint_filename, 'r') as f:

            for key in net.state_dict().keys():
                net.state_dict()[key] = np.asarray(f[key])


    current_lr = get_learning_rate(initial_epoch, config['learning_rate'])
    logging.info('initial learning rate: {}'.format(current_lr))
    optimizer = optim.SGD(net.parameters(), lr=current_lr, momentum=0.9, weight_decay=5e-4)
        
    with open(os.path.join(checkpoint_dir, 'training_log.csv'), 'a') as logfile:

        for epoch in range(initial_epoch, config['num_epochs']+1):

            new_lr = get_learning_rate(epoch, config['learning_rate'])
            if new_lr != current_lr:
                logging.info('learning rate changed to: {}'.format(new_lr))
                optimizer = optim.SGD(net.parameters(), lr=new_lr, momentum=0.9, weight_decay=5e-4)
                current_lr = new_lr
            
            logging.info('begin training epoch: {}'.format(epoch))
            train_log = train(net, epoch, loss_fn, optimizer, trainloader)
            
            logging.info('begin testing epoch: {}'.format(epoch))
            test_log = test(net, loss_fn, testloader)
            
            if epoch % 5 == 0 or epoch in [1,2,3,4,5]:
                checkpoint_filename = os.path.abspath(os.path.join(checkpoint_dir,
                                                   'checkpoint-epoch{}.h5'.format(str(epoch).zfill(5))))
                model_params = net.state_dict()
                with h5py.File(checkpoint_filename, 'w') as f:
                    for param_name in model_params.keys():
                        f.create_dataset(param_name, data=model_params[param_name])

                pytorch_checkpoint_filename = os.path.abspath(os.path.join(checkpoint_dir,
                                                   'pytorch_checkpoint-epoch{}.pt'.format(str(epoch).zfill(5))))
                torch.save(net, pytorch_checkpoint_filename)
                        
                logfile.write('{},{},{},{},{},{},{}\n'.format(epoch,
                                                            current_lr,
                                                            train_log['train_top1'],
                                                            train_log['train_loss'],
                                                            test_log['validation_top1'],
                                                            test_log['validation_loss'],
                                                            checkpoint_filename))
                logfile.flush()

                
def get_learning_rate(epoch, learning_rate_dict):

    num2key = {}
    for key in learning_rate_dict.keys():
        num2key[int(key)] = key

    nums = list(num2key.keys())
    nums = list(reversed(sorted(nums)))
    
    for i in range(len(nums)):
        if nums[i] <= epoch:
            return learning_rate_dict[num2key[nums[i]]]

    raise(Exception('Could not set learning rate for epoch:{} from lr_schedule:{}.'.format(
        epoch,
        json.dumps(learning_rate_dict, indent=4, sort_keys=True)
        )))


def check_for_required_params(d):

    keys = d.keys()
    assert 'data_hdf5' in keys
    assert 'network_name' in keys
    assert 'learning_rate' in keys
    assert 'batch_size' in keys
    assert 'num_epochs' in keys
    assert 'parameter_initialization' in keys

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('checkpoint_dir', type=str)
    parser.add_argument('--resume',  action='store_true')

    parser.add_argument('--json_config', type=str)

    args = parser.parse_args()

    if args.resume and args.json_config:
        parser.error('the "resume" and "json_config" args cannot be set at the same time.')
        
    if (args.resume is None) and (args.json_config is None):
        parser.error('either "resume" or "json_config" args must be set.')

    experiment_configuration = {}
        
    if args.resume:
        metadata_filename = os.path.join(args.checkpoint_dir, 'metadata.json')
        logging.info('loading parameters from checkpoint: {}.'.format(metadata_filename))

        with open(metadata_filename, 'r') as f:
            config = json.loads(f.read())
            
        check_for_required_params(config)
        keys = list(config.keys())
        assert 'creation_time' in keys
        assert 'data_hdf5' in keys
        assert 'checkpoint_dir' in keys
            
        for key in config.keys():
            experiment_configuration[key] = config[key]

            
    elif args.json_config:
        logging.info('reading configuration from json formatted config file: {}.'.format(args.json_config))
        
        with open(args.json_config) as f:
            config = json.loads(f.read())

        check_for_required_params(config)
            
        assert os.path.exists(config['data_hdf5']), 'the data_hdf5" field must be set and the data file must exist.'
        assert config['network_name'] in networks, 'the "network_name" field must be set and name a known network.'
        for key in config['learning_rate'].keys():
            lr = config['learning_rate'][key]
            assert lr > 0, 'the "learning_rate" field must be set and be greater than 0.'
        assert config['batch_size'] > 0, 'the "batch_size" field must be set and be greater than 0.'
        
        for key in config.keys():
            experiment_configuration[key] = config[key]


        if os.path.isdir(args.checkpoint_dir):
            logging.error('checkpoint dir already exists: {}'.format(args.checkpoint_dir))
            sys.exit(0)
    
        os.makedirs(args.checkpoint_dir)
            
        experiment_configuration['creation_time'] = time.strftime("%Y-%m-%d-%H:%M:%S", time.gmtime())
        experiment_configuration['checkpoint_dir'] = os.path.abspath(args.checkpoint_dir)
        
        with open(os.path.join(args.checkpoint_dir, 'metadata.json'), 'w') as f:
            f.write(json.dumps(experiment_configuration, indent=4, sort_keys=True))


        with open(os.path.join(args.checkpoint_dir, 'training_log.csv'), 'w') as logfile:

            logfile.write('# {\n')
            for k in experiment_configuration.keys():
                logfile.write('#     "{}": "{}",\n'.format(k, experiment_configuration[k]))
            logfile.write('# }\n')
            
            logfile.write('epoch, learning_rate, train_acc_top1, train_loss, validation_acc_top1, validation_loss, model_checkpoint\n') 
            logfile.flush()

            
    else:
        raise(Exception('neither "resume" nor "json_config" were set.'))

        
    logging.info('using the following configuration...')
    logging.info(json.dumps(experiment_configuration, indent=4, sort_keys=True))

    main(args.checkpoint_dir, args.resume, experiment_configuration)
