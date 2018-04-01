import torch
from torch.nn.modules.module import Module
from ..functions.noop import NoopEncoderFunc
from ..functions.noop import NoopDecoderFunc

from .._ext import mfc_wrapper

class NoopEncoder(Module):

    def __init__(self):
        super(NoopEncoder, self).__init__()

        self.mem1 = torch.FloatTensor()
        self.mem2 = torch.ByteTensor()

        
    def forward(self, inp, input_on_gpu=None):

        assert(input_on_gpu is not None)
        output = NoopEncoderFunc.apply(inp, self.mem1, self.mem2, input_on_gpu)

        return output

    
class NoopDecoder(Module):

    def __init__(self):
        super(NoopDecoder, self).__init__()

        self.mem1 = torch.FloatTensor()

        
    def forward(self, inp, put_output_on_gpu=None):

        assert(put_output_on_gpu is not None)
        output = NoopDecoderFunc.apply(inp, self.mem1, put_output_on_gpu)
                
        return output
    
