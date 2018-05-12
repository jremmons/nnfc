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
            compressed = encoder.forward(inputs)
            decompressed = decoder.forward(compressed)
            outputs = torch.from_numpy(decompressed)

            ctx.encoder = encoder
            ctx.decoder = decoder
            
            if on_gpu:
                outputs = outputs.cuda()
            
            return outputs

        
        @staticmethod
        def backward(ctx, grad_output):
            grad_output = ctx.decoder.backward(grad_output)
            grad_output = ctx.encoder.backward(grad_output)
            
            return grad_output, None, None

        
    def __init__(self, encoder_name='noop_encoder', decoder_name='noop_decoder',
                 encoder_params_dict={}, decoder_params_dict={}):
        super(CompressionLayer, self).__init__()

        self.encoder = nnfc_codec.EncoderContext(encoder_name, encoder_params_dict)
        self.decode = nnfc_codec.DecoderContext(decoder_name, decoder_params_dict)
        self.outputs = torch.FloatTensor()        

        
    def forward(self, inputs):

        outputs = CompressionLayer.CompressionLayerFunc.apply(inputs, self.encoder, self.decode)
        return outputs

