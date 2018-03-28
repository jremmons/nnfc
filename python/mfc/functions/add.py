# functions/add.py
import torch
from torch.autograd import Function
from .._ext import mfc_wrapper


class MyAddFunction(Function):
    def forward(self, input1, input2):
        output = input1.new()
        if not input1.is_cuda:
            mfc_wrapper.add_forward(input1, input2, output)
        else:
            mfc_wrapper.add_forward_cuda(input1, input2, output)
        return output

    def backward(self, grad_output):
        grad_input = grad_output.new()
        if not grad_output.is_cuda:
            mfc_wrapper.add_backward(grad_output, grad_input)
        else:
            mfc_wrapper.add_backward_cuda(grad_output, grad_input)
        return grad_input
