import argparse
import h5py
import os
import json
import shutil
import logging
import time
import timeit

from PIL import Image

import torch
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import numpy as np

import resnet
import mobilenetv2
import densenet
import dpn

NUM_EPOCHS = 200
logging.basicConfig(level=logging.DEBUG)

use_cuda = torch.cuda.is_available()

# TODO(jremmons) add functionality for restoring from checkpoint
# TODO(jremmons) add a better programmatic interface for defined network architecture

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
    

def train(model, loss_fn, optimizer, trainloader):

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

        train_loss += loss.data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(batch_labels.data.view_as(pred)).long().sum()

        loss.backward()
        optimizer.step()

        if batch_idx % 5 == 0:
            logging.info('Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                trainloader.batch_size * batch_idx, count,
                100. * trainloader.batch_size * batch_idx / count, loss.data[0]))

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

        test_loss += loss_fn(output, batch_labels).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(batch_labels.data.view_as(pred)).long().cpu().sum()
        
    logging.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, count,
        100. * correct / count))

    t2 = timeit.default_timer()
    logging.info('Test Epoch took {} seconds'.format(t2-t1))

    return {
        'validation_top1' : correct / count,
        'validation_loss' : test_loss
        }

    
def main(args):

    if os.path.isdir(args.checkpoint_dir):
        logging.error('checkpoint dir already exists: {}'.format(args.checkpoint_dir))
        return

    os.makedirs(args.checkpoint_dir)
    
    logging.info('loading data into memory')
    with h5py.File(args.data_hdf5, 'r') as f:
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = Cifar10(test_data_raw, test_data_labels, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    shutil.copy(args.data_hdf5, args.checkpoint_dir)

    metadata = {
        'creation_time' : time.strftime("%Y-%m-%d-%H:%M:%S", time.gmtime()),
        'data_hdf5' : os.path.abspath(args.data_hdf5),
        'checkpoint_dir' : os.path.abspath(args.checkpoint_dir),
        'batch_size' : args.batch_size,
        'netowrk_name' : args.net,
        'learning_rate' : args.lr,
        'training_epochs' : NUM_EPOCHS, 
        }
    
    with open(os.path.join(args.checkpoint_dir, 'metadata.json'), 'w') as f:
        f.write(json.dumps(metadata, indent=4, sort_keys=True))

    networks = {
        'resnet18' : resnet.ResNet18(),
        'resnet101' : resnet.ResNet101(),
        'mobilenetv2' : mobilenetv2.MobileNetV2(),
        'densenet121' : densenet.DenseNet121(),
        'densenet121' : densenet.DenseNet121(),
        'dpn92' : dpn.DPN92()
    }
        
    net = networks[args.net]
    if use_cuda:
        net.cuda()
    
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    #optimizer = optim.Adagrad(net.parameters(), lr=args.lr, lr_decay=0.01)
    #optimizer = optim.RMSprop(net.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    with open(os.path.join(args.checkpoint_dir, 'training_log.csv'), 'w') as logfile:

        logfile.write('# {\n')
        for k in metadata.keys():
            logfile.write('#     "{}": "{}",\n'.format(k, metadata[k]))
        logfile.write('# }\n')
        
        logfile.write('epoch, train_acc_top1, train_loss, validation_acc_top1, validation_loss, model_checkpoint\n') 
        logfile.flush()
        
        for epoch in range(1, NUM_EPOCHS+1):
            
            logging.info('begin training epoch: {}'.format(epoch))
            train_log = train(net, loss_fn, optimizer, trainloader)

            logging.info('begin testing epoch: {}'.format(epoch))
            test_log = test(net, loss_fn, testloader)

            if epoch % 5 == 0 or epoch in [1,2,3,4,5]:
                checkpoint_filename = os.path.abspath(os.path.join(args.checkpoint_dir,
                                                   'checkpoint-epoch{}.h5'.format(str(epoch).zfill(4))))
                model_params = net.state_dict()
                with h5py.File(checkpoint_filename, 'w') as f:
                    for param_name in model_params.keys():
                        f.create_dataset(param_name, data=model_params[param_name])

                logfile.write('{},{},{},{},{},{}\n'.format(epoch,
                                                           train_log['train_top1'],
                                                           train_log['train_loss'],
                                                           test_log['validation_top1'],
                                                           test_log['validation_loss'],
                                                           checkpoint_filename))
                logfile.flush()

                    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('data_hdf5', type=str)
    parser.add_argument('checkpoint_dir', type=str)

    # parser.add_argument('--compaction_factor', type=float, default=1.0)
    parser.add_argument('--net', type=str,
                        help='the name of the net to train.')                        
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=250,
                        help='train/test batch_size (default: 250)')
    # parser.add_argument('--resnet_blocks', type=str, default='[2,2,2,2]',
    #                     help='ex. [2,2,2,2]')
    
    args = parser.parse_args()
    main(args)

    


