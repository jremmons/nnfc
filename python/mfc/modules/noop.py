from torch.nn.modules.module import Module
from ..functions.noop import NoopEncoderFunc
from ..functions.noop import NoopDecoderFunc

class NoopEncoder(Module):
    def forward(self, inp):
        return NoopEncoderFunc()(inp)

class NoopDecoder(Module):
    def forward(self, inp):
        return NoopDecoderFunc()(inp)
    
