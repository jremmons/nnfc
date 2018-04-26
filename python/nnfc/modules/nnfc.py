import torch
from torch.autograd import Variable 
from torch.nn.modules.module import Module
from torch.autograd import Function

from .._ext import nnfc_codec

class CompressionLayer(Module):

    class CompressionLayerFunc(Function):
        
        @staticmethod
        def forward(ctx, inputs, encoder, decoder):

            on_gpu = inputs.is_cuda
            
            inputs = inputs.cpu().numpy()
            compress_decompress = decoder.decode( encoder.encode(inputs) )
            outputs = torch.from_numpy(compress_decompress)

            output = outputs + 1
            
            if on_gpu:
                outputs = outputs.cuda()
            
            return outputs
        
        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, None, None

        
    def __init__(self, codec_name='noop'):
        super(CompressionLayer, self).__init__()

        self.encoder = nnfc_codec.EncoderContext(codec_name)
        self.decode = nnfc_codec.DecoderContext(codec_name)

        self.outputs = torch.FloatTensor()        

        
    def forward(self, inputs):

        # compressed, encoder_states = self.encoder.encode(inputs, states=None)

        # outputs, states = self.decoder.decode(inputs, states=None)
        
        # assert(input_on_gpu is not None)
        # outputs, compressed_binary_blobs, states = self.codec.encode_and_decode(inp,
        #                                                                         states=None,
        #                                                                         output_buffer=None)

        # compressed_binary_blobs: is a python list of buffers corresponding to the compressed video
        # states: is a python list of opaque buffers that are serialized versions of the encoder & decoder states
        # output_buffer: should be the same size as the original input

        outputs = CompressionLayer.CompressionLayerFunc.apply(inputs, self.encoder, self.decode)
                
        return outputs

