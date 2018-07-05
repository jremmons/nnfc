import torch
import zlib
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.autograd import Function

from .._ext import nnfc_codec

class CompressionLayer(Module):

    class CompressionLayerFunc(Function):

        @staticmethod
        def forward(ctx, inputs, encoder, decoder, statistics):

            on_gpu = inputs.is_cuda

            if on_gpu:
                inputs = inputs.cpu().numpy()
            else:
                inputs = inputs.detach().numpy()

            compressed = encoder.forward(inputs)

            # compressed_0 = [zlib.compress(x.tobytes()) for x in compressed]
            # statistics['sizeof_intermediates'] = list(map(lambda x: (len(x),), compressed_0))
            statistics['sizeof_intermediates'] = list(map(lambda x: x.shape, compressed))

            decompressed = decoder.forward(compressed)
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
