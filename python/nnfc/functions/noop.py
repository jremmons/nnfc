import torch
from torch.autograd import Function

class NoopEncoderFunc(Function):

    @staticmethod
    def forward(ctx, inp):
        return inp

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

    
class NoopDecoderFunc(Function):

    @staticmethod
    def forward(ctx, inp):
        return inp
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
