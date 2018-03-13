import argparse
import h5py
import logging
import resnet
import timeit

import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

NUM_EPOCHS = 1000
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

    t1 = timeit.default_timer()
    for i in range(0, data_len, batch_size):

        batch_data = (data[i:i+batch_size, :, :, :].astype(np.float32) / 128.0) - 1
        batch_data = torch.autograd.Variable(torch.from_numpy(batch_data)).cuda()

        batch_labels = torch.autograd.Variable(torch.from_numpy(data_labels[i:i+batch_size].astype(np.int64))).cuda()
        
        optimizer.zero_grad()
        output = model(batch_data)
        loss = loss_fn(output, batch_labels)
        loss.backward()
        optimizer.step()

        if i % (batch_size) == 0 and i != 0:
            logging.info('Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                i, data_len,
                100. * i / data_len, loss.data[0]))

    t2 = timeit.default_timer()
    logging.info('Train Epoch took {} seconds'.format(t2-t1))
    
    
def test(model, loss_fn, batch_size, data, data_labels):

    model.eval()

    data_len = len(data)
    data_labels_len = len(data_labels)
    assert data_len == data_labels_len, 'data_len != data_labels_len; {} != {}'.format(data_len, data_labels_len)

    test_loss = 0
    correct = 0
    t1 = timeit.default_timer()
    for i in range(0, data_len, batch_size):

        batch_data = (data[i:i+batch_size, :, :, :].astype(np.float32) / 128.0) - 1
        batch_data = torch.autograd.Variable(torch.from_numpy(batch_data)).cuda()
        batch_labels = torch.autograd.Variable(torch.from_numpy(data_labels[i:i+batch_size].astype(np.int64))).cuda()
        
        output = model(batch_data)
        test_loss += loss_fn(output, batch_labels).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(batch_labels.data.view_as(pred)).long().cpu().sum()
        
    logging.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, data_len,
        100. * correct / data_len))

    t2 = timeit.default_timer()
    logging.info('Test Epoch took {} seconds'.format(t2-t1))
        
def main(args):

    logging.info('loading data into memory')
    with h5py.File(args.data_hdf5, 'r') as f:
        train_data_raw = np.asarray(f['train_data_raw'])
        train_data_labels = np.asarray(f['train_data_labels'])
        test_data_raw = np.asarray(f['test_data_raw'])
        test_data_labels = np.asarray(f['test_data_labels'])
    logging.info('done! (loading data into memory)')

    
    net = torch.nn.DataParallel(resnet.ResNet18())
    net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, NUM_EPOCHS+1):

        logging.info('begin training epoch: {}'.format(epoch))
        train(net, loss_fn, optimizer, args.batch_size, train_data_raw, train_data_labels)
        
        logging.info('begin testing epoch: {}'.format(epoch))
        test(net, loss_fn, args.batch_size, test_data_raw, test_data_labels)

        # save model params here
        
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('data_hdf5', type=str)
    parser.add_argument('temp_dir', type=str)

    parser.add_argument('--lr', type=float, default=0.001, 
                        help='learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='train/test batch_size (default: 1000)')
    
    args = parser.parse_args()
    main(args)

    


