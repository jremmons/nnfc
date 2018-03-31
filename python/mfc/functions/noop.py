# functions/add.py
import torch
from torch.autograd import Function
from .._ext import mfc_wrapper

import timeit
import sys

class NoopEncoderFunc(Function):

    @staticmethod
    def forward(ctx, inp, mem1, mem2):

        t1 = timeit.default_timer()
        inp_cpu = inp
        if inp_cpu.is_cuda:
            mfc_wrapper.device_to_host_copy(mem1, inp)
            inp_cpu = mem1
        t2 = timeit.default_timer()
        #sys.stderr.write('copy to host time: {}\n'.format(t2-t1))

        t1 = timeit.default_timer()
        mfc_wrapper.noop_encode_forward(inp_cpu, mem2)
        t2 = timeit.default_timer()
        #sys.stderr.write('encode time: {}\n'.format(t2-t1))

        return mem2

    @staticmethod
    def backward(ctx, grad_output):
        #grad_input = grad_output.new()
        #mfc_wrapper.noop_encode_backward(grad_output, grad_input)
        return grad_output, None, None

    
class NoopDecoderFunc(Function):

    @staticmethod
    def forward(ctx, inp, mem1):

        t1 = timeit.default_timer()
        inp = inp.cpu()
        mfc_wrapper.noop_decode_forward(inp, mem1)
        t2 = timeit.default_timer()
        #sys.stderr.write('decode time: {}\n'.format(t2-t1))

        t1 = timeit.default_timer()
        output_cuda = mem1.cuda()
        t2 = timeit.default_timer()
        #sys.stderr.write('copy to gpu time: {}\n\n'.format(t2-t1))
        
        return output_cuda

    
    @staticmethod
    def backward(ctx, grad_output):
        #grad_input = grad_output.new()
        #mfc_wrapper.noop_decode_backward(grad_output, grad_input)
        return grad_output, None
    
