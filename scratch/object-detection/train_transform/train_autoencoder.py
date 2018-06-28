#!/usr/bin/env python3

import os
import h5py
import sys
import pprint

import numpy as np
import multiprocessing as mp

import torch
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

from PIL import Image

import utils
import timeit
import yolov3_dn1 as yolo
import yolov3_ae as yolo_ae

EPOCHS = 150
CHECKPOINT_PREFIX = 'autoencoder_{}.h5'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_labels(label, size):
    labels = []
    for line in label.split("\n")[:-1]:
        data = line.strip().split(' ')
        bb_o = [float(x) for x in data[1:]]
        idx = int(data[0])

        bb = [0] * 4
        bb[0] = size * (bb_o[0] - bb_o[2] / 2)
        bb[1] = size * (bb_o[1] - bb_o[3] / 2)
        bb[2] = size * (bb_o[0] + bb_o[2] / 2)
        bb[3] = size * (bb_o[1] + bb_o[3] / 2)

        labels += [{
            'coco_idx': idx,
            'bb': bb
        }]

    return labels

class YoloDataset(Dataset):
    def __init__(self, images_path, labels_path, transforms, size=416):
        self.images = []
        self.labels = []
        self.images_path = images_path
        self.labels_path = labels_path
        self.size = size
        self.max_items = 50

        self.transforms = transforms

        for img in sorted(os.listdir(images_path)):
            lbl = os.path.basename(img).split('.')[0] + '.txt'
            label_path = os.path.join(labels_path, lbl)

            if os.path.exists(label_path):
                self.images += [img]
                self.labels += [lbl]

        assert(len(self.images) == len(self.labels))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.images_path, self.images[index])
        label_path = os.path.join(self.labels_path, self.labels[index])

        labels = []
        with open(label_path) as label_file:
            labels += [label_file.read()]

        image_original = Image.open(image_path).convert('RGB')
        image_original = image_original.resize((self.size, self.size))
        image = image_original
        #image = np.asarray(image_original)
        
        image = self.transforms(image)

        return image, labels

def main(images_path, labels_path, save_dir):
    
    save_dir = os.path.abspath(save_dir)
    if os.path.exists(save_dir):
        raise Exception('save_dir already exists! ' + save_dir)
    else:
        os.makedirs(save_dir)

    size = 416

    t = transforms.Compose([
        transforms.RandomCrop(size, padding=52),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data = DataLoader(YoloDataset(images_path, labels_path, t, size), batch_size=16, shuffle=True, num_workers=mp.cpu_count())
    model = yolo.load_model()
    model.to(device)
    model.eval()

    #autoencoder = yolo_ae.AutoEncoderI()
    #autoencoder = yolo_ae.AutoEncoder0()
    #autoencoder = yolo_ae.AutoEncoder1()
    autoencoder.to(device)
    autoencoder.train()

    mse_loss_fn = torch.nn.MSELoss()
    learning_rate_schedule = \
    {
        '5' : 0.001,
        '10' : 0.0001,
        '20' : 0.000001,
    }

    def select_lr(epoch):

        lr_keys = sorted(list(map(int, learning_rate_schedule.keys())))
        for key in lr_keys:
            if epoch < key:
                return learning_rate_schedule[str(key)]

        return None
            
    prev_lr = select_lr(0)
    optimizer = optim.Adam(autoencoder.parameters(), lr=prev_lr)
    for epoch_num in range(EPOCHS):

        total_loss = 0

        lr = select_lr(epoch_num)
        print('learning rate:', lr)

        if prev_lr != lr:
            print('learning rate changed to:', lr, 'from:', prev_lr)
            optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
            prev_lr = lr
            
        for i, (local_batch, local_labels) in enumerate(data):

            local_batch = local_batch.to(device)
            targets = torch.autograd.Variable(model(local_batch).detach())

            output = autoencoder(targets)

            loss = mse_loss_fn(output, targets)
            total_loss += loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print('minibatch loss: {}'.format(loss))
            

        print('epoch completed. total_loss: {}'.format(total_loss))
        print('saving checkpoint')
        checkpoint_filename = os.path.join(save_dir, CHECKPOINT_PREFIX.format(epoch_num))
        with h5py.File(checkpoint_filename, 'w') as f:

            autoencoder_params = autoencoder.state_dict()
            for key in autoencoder_params.keys():
                f.create_dataset(key, data=autoencoder_params[key])
            
if __name__ == '__main__':
    main(images_path=sys.argv[1], labels_path=sys.argv[2], save_dir=sys.argv[3])
