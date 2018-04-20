#include <TH/TH.h>
#include <stdlib.h>

#include <cassert>
#include <cstdint>
#include <iostream>

#include "blobs.hh"
#include "common.hh"
#include "nnfc.hh"
    
// functions must 'extern "C"' in order to be callable from within pytorch/python
// https://github.com/torch/TH/blob/master/generic/THTensor.h

extern "C" int nnfc_encode_forward(THFloatTensor *input, THByteTensor *output) {
    TorchFloatBlob4D input_blob{input};
    TorchByteBlob1D output_blob{output};

    NNFC::encode(input_blob, output_blob);
    
    return _TORCH_SUCCESS;    
}


extern "C" int nnfc_encode_backward(THFloatTensor *grad_output, THFloatTensor *grad_input) {

    THFloatTensor_resizeAs(grad_input, grad_output);
    THFloatTensor_fill(grad_input, 1);
    return _TORCH_SUCCESS;
}


extern "C" int nnfc_decode_forward(THByteTensor *input, THFloatTensor *output) {

    TorchByteBlob1D input_blob{input};
    TorchFloatBlob4D output_blob{output};

    NNFC::decode(input_blob, output_blob);
    
    return _TORCH_SUCCESS;    
}


extern "C" int nnfc_decode_backward(THFloatTensor *grad_output, THFloatTensor *grad_input) {

    THFloatTensor_resizeAs(grad_input, grad_output);
    THFloatTensor_fill(grad_input, 1);
    return _TORCH_SUCCESS;
}
