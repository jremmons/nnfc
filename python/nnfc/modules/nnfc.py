import torch
from torch.nn.modules.module import Module
from ..functions.nnfc import NnfcEncoderFunc
from ..functions.nnfc import NnfcDecoderFunc

from .._ext import nnfc_wrapper

class NnfcEncoder(Module):

    def __init__(self):
        super(NnfcEncoder, self).__init__()

        self.mem1 = torch.FloatTensor()
        self.mem2 = torch.ByteTensor()

        
    def forward(self, inp, input_on_gpu=None):

        assert(input_on_gpu is not None)
        output = NnfcEncoderFunc.apply(inp, self.mem1, self.mem2, input_on_gpu)

        return output

    
class NnfcDecoder(Module):

    def __init__(self):
        super(NnfcDecoder, self).__init__()

        self.mem1 = torch.FloatTensor()

        
    def forward(self, inp, put_output_on_gpu=None):

        assert(put_output_on_gpu is not None)
        output = NnfcDecoderFunc.apply(inp, self.mem1, put_output_on_gpu)
                
        return output
    
