import torch
import torch.nn as nn
from torch.autograd import Variable
from my_package.modules.add import MyAddModule

import timeit

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.add = MyAddModule()

    def forward(self, input1, input2):
        out = self.add(input1, input1)
        return out
        
model = MyNetwork()
x = torch.range(1, 7*8).view(7, 8)
input1, input2 = Variable(x), Variable(x * 4)

out = model(input1, input2)

# N = 10000
# times = []
# for _ in range(100):

#     t1 = timeit.default_timer()
#     out = model(input1, input2)
#     t2 = timeit.default_timer()
#     model_runtime = t2 - t1
#     #print('model run time: {}'.format(model_runtime))

# for _ in range(N):

#     t1 = timeit.default_timer()
#     out = model(input1, input2)
#     t2 = timeit.default_timer()
#     model_runtime = t2 - t1
#     #print('model run time: {}'.format(model_runtime))

#     t1 = timeit.default_timer()
#     out = input1 + input2
#     t2 = timeit.default_timer()
#     numpy_add_runtime = t2-t1
#     #print('numpy add run time: {}'.format(t2-t1))

#     #times.append(model_runtime - numpy_add_runtime)
#     times.append(model_runtime)
    
# print(sum(times) / N)
    
# if torch.cuda.is_available():
#     input1, input2, = input1.cuda(), input2.cuda()
#     print(model(input1, input2))
#     print(input1 + input2)
