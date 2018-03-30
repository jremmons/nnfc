# functions/add.py
import torch
from torch.autograd import Function
from .._ext import mfc_wrapper

class NoopEncoderFunc(Function):

    def forward(self, inp):
        output = torch.ByteTensor() # create reference that will be filled
        mfc_wrapper.noop_encode_forward(inp, output)

        return output

    def backward(self, grad_output):
        #grad_input = grad_output.new()
        #mfc_wrapper.noop_encode_backward(grad_output, grad_input)
        return grad_output

    
class NoopDecoderFunc(Function):

    def forward(self, inp):
        output = torch.FloatTensor() # create reference that will be filled
        mfc_wrapper.noop_decode_forward(inp, output)

        return output

    def backward(self, grad_output):
        #grad_input = grad_output.new()
        #mfc_wrapper.noop_decode_backward(grad_output, grad_input)
        return grad_output
    
