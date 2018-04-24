import torch
from torch.autograd import Function

import timeit
import sys

class NoopEncoderFunc(Function):

    @staticmethod
    def forward(ctx, inp, mem1, mem2, gpu):
        return inp

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

    
class NoopDecoderFunc(Function):

    @staticmethod
    def forward(ctx, inp, mem1, gpu):
        return inp
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None
    
