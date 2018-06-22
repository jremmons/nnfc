import os

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

from PIL import Image

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
        image = np.asarray(image_original)

        image = self.transforms(image)

        return image, labels
