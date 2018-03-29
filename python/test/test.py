import torch
import torch.nn as nn
from torch.autograd import Variable
from mfc.modules.noop import Noop

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.noop = Noop()

    def forward(self, inp):
        return self.noop(inp)

model = MyNetwork()

x = torch.arange(0, 30000).view(3, 100, 100)
#x = torch.arange(0, 10000).view(100, 100)

inp = Variable(x)
print(model(inp))
