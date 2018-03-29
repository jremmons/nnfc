import torch
import torch.nn as nn
from torch.autograd import Variable

from mfc.modules.noop import NoopEncoder
from mfc.modules.noop import NoopDecoder


class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.noop_encoder = NoopEncoder()
        self.noop_decoder = NoopDecoder()

    def forward(self, inp):
        encoded = self.noop_encoder(inp)
        decoded = self.noop_decoder(encoded)
        return decoded

model = MyNetwork()

x = torch.arange(0, 30000).view(3, 100, 100) % 256

inp = Variable(x)

print(inp)
out = model(inp)
print(out)

print('noop success:', (inp == out).all())
