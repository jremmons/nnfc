import numpy as np

import torch
import torch.utils.data

from PIL import Image
from nnfc.modules.nnfc import CompressionLayer

class Cifar10(torch.utils.data.Dataset):

    def __init__(self, data_raw, data_labels, transform=None):

        r = data_raw[:,0,:,:]
        g = data_raw[:,1,:,:]
        b = data_raw[:,2,:,:]

        self.data_raw = np.stack([r,g,b], axis=-1)
        self.data_labels = data_labels
        self.transform = transform

        # self.nnfc_compression_layer = CompressionLayer(encoder_name='jpeg_image_encoder',
        #                                                encoder_params_dict={'quantizer' : 100},
        #                                                decoder_name='jpeg_image_decoder',
        #                                                decoder_params_dict={})

    def __len__(self):

        return len(self.data_raw)

    def __getitem__(self, idx):

        image = self.data_raw[idx,:,:,:]
        # image = image.transpose((2,0,1))
        # image = image.astype(np.float32)
        # image = np.expand_dims(image, axis=0)
        # image = np.ascontiguousarray(image)
        
        # image = torch.from_numpy(image)
        # image = self.nnfc_compression_layer(image)
        # image = image.numpy()
        
        # image = np.squeeze(image, axis=0)
        # image = image.astype(np.uint8)
        # image = image.transpose((1,2,0))
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, self.data_labels[idx].astype(np.int64)

    
def check_for_required_params(d):

    keys = d.keys()
    assert 'data_hdf5' in keys
    assert 'network_name' in keys
    assert 'learning_rate' in keys
    assert 'batch_size' in keys
    assert 'num_epochs' in keys
    assert 'parameter_initialization' in keys


