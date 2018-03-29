from torch.nn.modules.module import Module
from ..functions.noop import NoopFunc

class Noop(Module):
    def forward(self, inp):
        return NoopFunc()(inp)
