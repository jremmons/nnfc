# functions/add.py
import torch
from torch.autograd import Function
from .._ext import mfc_wrapper

class NoopFunc(Function):
    def forward(self, inp):
        output = inp.new()
        mfc_wrapper.add_forward(inp, output)
        return output

    def backward(self, grad_output):
        grad_input = grad_output.new()
        mfc_wrapper.add_backward(grad_output, grad_input)
        return grad_input
