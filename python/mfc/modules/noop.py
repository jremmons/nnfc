import torch
from torch.nn.modules.module import Module
from ..functions.noop import NoopEncoderFunc
from ..functions.noop import NoopDecoderFunc

from .._ext import mfc_wrapper

class NoopEncoder(Module):

    def __init__(self):
        super(NoopEncoder, self).__init__()

        self.mem1 = None
        self.mem2 = None

        
    def __del__(self):

        if self.mem1 is not None:
            mfc_wrapper.free_pinned_tensor_float(self.mem1)
            
        if self.mem2 is not None:
            mfc_wrapper.free_pinned_tensor_byte(self.mem2)

            
    def forward(self, inp):

        # TODO(jremmons) check shape too!
        if self.mem1 is None:
            self.mem1 = torch.FloatTensor(500,128,16,16)
            mfc_wrapper.alloc_pinned_tensor_float(self.mem1)

        if self.mem2 is None:
            self.mem2 = torch.ByteTensor(4*500*16*16*500 + 40)
            mfc_wrapper.alloc_pinned_tensor_byte(self.mem2)

        return NoopEncoderFunc.apply(inp, self.mem1, self.mem2)

    
class NoopDecoder(Module):

    def __init__(self):
        super(NoopDecoder, self).__init__()

        self.mem1 = None

    def __del__(self):

        if self.mem1 is not None:
            mfc_wrapper.free_pinned_tensor_float(self.mem1)

            
    def forward(self, inp):

        if self.mem1 is None:
            self.mem1 = torch.FloatTensor(500,128,16,16)
            mfc_wrapper.alloc_pinned_tensor_float(self.mem1)

        return NoopDecoderFunc.apply(inp, self.mem1)
    
