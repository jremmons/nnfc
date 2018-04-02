import argparse
import h5py
import os
import json
import shutil
import logging
import resnet
import time
import timeit

import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

NUM_EPOCHS = 100
logging.basicConfig(level=logging.DEBUG)


def train(model, loss_fn, optimizer, batch_size, _data, _data_labels):

    model.train()

    data_len = len(_data)
    data_labels_len = len(_data_labels)
    assert data_len == data_labels_len, 'data_len != data_labels_len; {} != {}'.format(data_len, data_labels_len)

    # shuffle the data and labels in the same way
    p = np.random.permutation(len(_data))
    data = _data[p]
    data_labels = _data_labels[p]


    train_loss = 0
    correct = 0
    t1 = timeit.default_timer()
    for i in range(0, data_len, batch_size):

        batch_data = data[i:i+batch_size, :, :, :]
        batch_data = torch.autograd.Variable(torch.from_numpy(batch_data))#.cuda()

        batch_labels = torch.autograd.Variable(torch.from_numpy(data_labels[i:i+batch_size].astype(np.int64)))#.cuda()
        
        optimizer.zero_grad()
        output = model(batch_data)
        loss = loss_fn(output, batch_labels)

        train_loss += loss.data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(batch_labels.data.view_as(pred)).long().sum()

        loss.backward()
        optimizer.step()

        if i % (batch_size) == 0 and i != 0:
            logging.info('Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                i, data_len,
                100. * i / data_len, loss.data[0]))

    t2 = timeit.default_timer()
    logging.info('Train Epoch took {} seconds'.format(t2-t1))

    return {
        'train_top1' : correct / data_len,
        'train_loss' : train_loss
        }
    
    
def test(model, loss_fn, batch_size, data, data_labels):

    model.eval()

    data_len = len(data)
    data_labels_len = len(data_labels)
    assert data_len == data_labels_len, 'data_len != data_labels_len; {} != {}'.format(data_len, data_labels_len)

    test_loss = 0
    correct = 0
    t1 = timeit.default_timer()
    for i in range(0, data_len, batch_size):

        batch_data = data[i:i+batch_size, :, :, :]
        batch_data = torch.autograd.Variable(torch.from_numpy(batch_data))#.cuda()
        # batch_labels = torch.autograd.Variable(torch.from_numpy(data_labels[i:i+batch_size].astype(np.int64)))#.cuda()

        t1 = timeit.default_timer()
        output = model(batch_data)
        t2 = timeit.default_timer()
        print('fwd:', t2-t1)

        continue

        test_loss += loss_fn(output, batch_labels).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(batch_labels.data.view_as(pred)).long().cpu().sum()
        
    logging.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, data_len,
        100. * correct / data_len))

    t2 = timeit.default_timer()
    logging.info('Test Epoch took {} seconds'.format(t2-t1))

    return {
        'validation_top1' : correct / data_len,
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


    logging.info('compute and subtract train_data mean pixel value; squash to 0-1 as well')
    train_data_raw_mean = np.mean(train_data_raw, axis=0)

    train_data_raw = (train_data_raw.astype(np.float64) - train_data_raw_mean) / 255.0
    test_data_raw = (test_data_raw.astype(np.float64) - train_data_raw_mean) / 255.0

    train_data_raw = train_data_raw.astype(np.float32)
    test_data_raw = test_data_raw.astype(np.float32)
    logging.info('done! (compute and subtract train_data mean pixel value; squash to 0-1 as well)')
    
    shutil.copy(args.data_hdf5, args.checkpoint_dir)

    metadata = {
        'creation_time' : time.strftime("%Y-%m-%d-%H:%M:%S", time.gmtime()),
        'data_hdf5' : os.path.abspath(args.data_hdf5),
        'checkpoint_dir' : os.path.abspath(args.checkpoint_dir),
        'batch_size' : args.batch_size,
        'resnet_blocks' : args.resnet_blocks,
        'compaction_factor' : args.compaction_factor,
        'learning_rate' : args.lr,
        'training_epochs' : NUM_EPOCHS, 
        }
    
    with open(os.path.join(args.checkpoint_dir, 'metadata.json'), 'w') as f:
        f.write(json.dumps(metadata, indent=4, sort_keys=True))
    
    # resnet-18
    resnet_blocks = [2,2,2,2]

    if args.resnet_blocks:
        resnet_blocks = eval(args.resnet_blocks)

    logging.info('Using the following resnet block architecture: {}'.format(resnet_blocks))
    net = torch.nn.DataParallel(resnet.AutoencoderResNet(compaction_factor=args.compaction_factor, num_blocks=resnet_blocks))
    # net.cuda()

    # resnet-34 
    #resnet34_blocks = [3,4,6,3]
    #net = torch.nn.DataParallel(resnet.AutoencoderResNet(compaction_factor=args.compaction_factor, num_blocks=resnet34_blocks))
    #net.cuda()

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
            
            # logging.info('begin training epoch: {}'.format(epoch))
            # train_log = train(net, loss_fn, optimizer, args.batch_size, train_data_raw, train_data_labels)

            logging.info('begin testing epoch: {}'.format(epoch))
            test_log = test(net, loss_fn, args.batch_size, test_data_raw, test_data_labels)

            # if epoch % 5 == 0 or epoch in [1,2,3,4,5]:
            #     checkpoint_filename = os.path.abspath(os.path.join(args.checkpoint_dir,
            #                                        'checkpoint-epoch{}.h5'.format(str(epoch).zfill(4))))
            #     model_params = net.state_dict()
            #     with h5py.File(checkpoint_filename, 'w') as f:
            #         for param_name in model_params.keys():
            #             f.create_dataset(param_name, data=model_params[param_name])

            #     logfile.write('{},{},{},{},{},{}\n'.format(epoch,
            #                                                train_log['train_top1'],
            #                                                train_log['train_loss'],
            #                                                test_log['validation_top1'],
            #                                                test_log['validation_loss'],
            #                                                checkpoint_filename))
            #     logfile.flush()

                    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('data_hdf5', type=str)
    parser.add_argument('checkpoint_dir', type=str)

    parser.add_argument('--compaction_factor', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='train/test batch_size (default: 1000)')
    parser.add_argument('--resnet_blocks', type=str, default='[2,2,2,2]',
                        help='ex. [2,2,2,2]')
    
    args = parser.parse_args()
    main(args)

    


