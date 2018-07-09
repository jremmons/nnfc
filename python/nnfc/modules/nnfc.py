import time
import zlib

import numpy as np
import torch
import zlib
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.autograd import Function

from .._ext import nnfc_codec

BLOCK_SIZE = (32, 13, 13)
TRANSFORM = np.load("/home/sadjad/projects/nnfc-temp/klt/transform.npy")
MEANS = np.load("/home/sadjad/projects/nnfc-temp/klt/mean.npy").reshape(-1)

def forward_transform_block(block):
    d = np.dot(TRANSFORM.T, block.reshape(-1) - MEANS)
    d[len(d)//5:] = 0.0
    return d.reshape(*block.shape)

def backward_transform_block(block):
    return (np.dot(TRANSFORM, block.reshape(-1)) + MEANS).reshape(*block.shape)

class CompressionLayer(Module):

    class CompressionLayerFunc(Function):

        @staticmethod
        def forward(ctx, inputs, encoder, decoder, statistics):
            on_gpu = inputs.is_cuda

            if on_gpu:
                inputs = inputs.cpu().numpy()
            else:
                inputs = inputs.detach().numpy()

            #np.save('/home/sadjad/projects/nnfc-temp/klt/activations/%d' % (time.time() * 1000), inputs)

            for h in range(inputs.shape[0]):
                for i in range(inputs.shape[1] // BLOCK_SIZE[0]):
                    for j in range(inputs.shape[2] // BLOCK_SIZE[1]):
                        for k in range(inputs.shape[3] // BLOCK_SIZE[2]):
                            block = inputs[h, i*BLOCK_SIZE[0]:(i+1)*BLOCK_SIZE[0],
                                              j*BLOCK_SIZE[1]:(j+1)*BLOCK_SIZE[1],
                                              k*BLOCK_SIZE[2]:(k+1)*BLOCK_SIZE[2]]
                            inputs[h, i*BLOCK_SIZE[0]:(i+1)*BLOCK_SIZE[0],
                                      j*BLOCK_SIZE[1]:(j+1)*BLOCK_SIZE[1],
                                      k*BLOCK_SIZE[2]:(k+1)*BLOCK_SIZE[2]] = \
                                forward_transform_block(block)

            compressed = encoder.forward(inputs)
            compressed_0 = [zlib.compress(x.tobytes(), level=9) for x in compressed]

            statistics['sizeof_intermediates'] = list(map(lambda x: (len(x),), compressed_0))
            # statistics['sizeof_intermediates'] = list(map(lambda x: x.shape, compressed))

            decompressed = decoder.forward(compressed)

            for h in range(inputs.shape[0]):
                for i in range(inputs.shape[1] // BLOCK_SIZE[0]):
                    for j in range(inputs.shape[2] // BLOCK_SIZE[1]):
                        for k in range(inputs.shape[3] // BLOCK_SIZE[2]):
                            block = decompressed[h, i*BLOCK_SIZE[0]:(i+1)*BLOCK_SIZE[0],
                                                    j*BLOCK_SIZE[1]:(j+1)*BLOCK_SIZE[1],
                                                    k*BLOCK_SIZE[2]:(k+1)*BLOCK_SIZE[2]]
                            decompressed[h, i*BLOCK_SIZE[0]:(i+1)*BLOCK_SIZE[0],
                                            j*BLOCK_SIZE[1]:(j+1)*BLOCK_SIZE[1],
                                            k*BLOCK_SIZE[2]:(k+1)*BLOCK_SIZE[2]] = \
                                backward_transform_block(block)

            outputs = torch.from_numpy(decompressed)

            ctx.encoder = encoder
            ctx.decoder = decoder

            if on_gpu:
                outputs = outputs.cuda()

            return outputs


        @staticmethod
        def backward(ctx, grad_output):
            # grad_output = ctx.decoder.backward(grad_output)
            # grad_output = ctx.encoder.backward(grad_output)

            return grad_output, None, None, None


    def __init__(self, encoder_name='noop_encoder', decoder_name='noop_decoder',
                 encoder_params_dict={}, decoder_params_dict={}):
        super(CompressionLayer, self).__init__()

        self.encoder = nnfc_codec.EncoderContext(encoder_name, encoder_params_dict)
        self.decoder = nnfc_codec.DecoderContext(decoder_name, decoder_params_dict)

        self.running_stats = {}


    def get_compressed_sizes(self):

        return self.running_stats['sizeof_intermediates']


    def forward(self, inputs):

        outputs = CompressionLayer.CompressionLayerFunc.apply(inputs, self.encoder, self.decoder, self.running_stats)
        return outputs
