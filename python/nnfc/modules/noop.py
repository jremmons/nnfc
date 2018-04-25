import torch
from torch.nn.modules.module import Module
from ..functions.noop import NoopEncoderFunc
from ..functions.noop import NoopDecoderFunc

class NoopEncoder(Module):

    def __init__(self):
        super(NoopEncoder, self).__init__()

        
    def forward(self, inp):
        return NoopEncoderFunc.apply(inp)

    
class NoopDecoder(Module):

    def __init__(self):
        super(NoopDecoder, self).__init__()

        
    def forward(self, inp):
        return NoopDecoderFunc.apply(inp)
    
