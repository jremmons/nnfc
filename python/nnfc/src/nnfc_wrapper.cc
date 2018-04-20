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

extern "C" int nnfc_encode_forward(THFloatTensor *input, THByteTensor *output)
{

    // sanity checking
    {
        int input_contiguous = THFloatTensor_isContiguous(input);
        WrapperAssert(input_contiguous, "input array not contiguous!");
        int input_ndims = THFloatTensor_nDimension(input);
        WrapperAssert(input_ndims == 4, "input dimensions must be 4");
    }

    // munge the blobs
    Eigen::Index n_size = THFloatTensor_size(input, 0);
    Eigen::Index c_size = THFloatTensor_size(input, 1);
    Eigen::Index h_size = THFloatTensor_size(input, 2);
    Eigen::Index w_size = THFloatTensor_size(input, 3);
    float* input_data = THFloatTensor_data(input);
    TorchFloatBlob4D input_blob{input, input_data, n_size, c_size, h_size, w_size};
    
    Eigen::Index b_size = THByteTensor_nElement(output);
    uint8_t *output_data = THByteTensor_data(output);
    TorchByteBlob1D output_blob{output, output_data, b_size};

    // call the encoder
    NNFC::encode(input_blob, output_blob);
    
    return _TORCH_SUCCESS;    
}


extern "C" int nnfc_encode_backward(THFloatTensor *grad_output, THFloatTensor *grad_input)
{
    THFloatTensor_resizeAs(grad_input, grad_output);
    THFloatTensor_fill(grad_input, 1);
    return _TORCH_SUCCESS;
}


extern "C" int nnfc_decode_forward(THByteTensor *input, THFloatTensor *output)
{

    // sanity checking
    {
        int input_contiguous = THByteTensor_isContiguous(input);
        WrapperAssert(input_contiguous, "input array not contiguous!");
        
        int input_ndims = THByteTensor_nDimension(input);
        WrapperAssert(input_ndims == 1, "input dimensions must be 1");
    }
    
    // munge the blobs
    Eigen::Index b_size = THByteTensor_nElement(input);
    uint8_t* input_data = THByteTensor_data(input);
    TorchByteBlob1D input_blob{input, input_data, b_size};
    
    Eigen::Index n_size = THFloatTensor_nElement(output);
    float* output_data = THFloatTensor_data(output);
    TorchFloatBlob4D output_blob{output, output_data, n_size, 1, 1, 1};

    // call the decoder
    NNFC::decode(input_blob, output_blob);
    
    return _TORCH_SUCCESS;    
}


extern "C" int nnfc_decode_backward(THFloatTensor *grad_output, THFloatTensor *grad_input)
{
    THFloatTensor_resizeAs(grad_input, grad_output);
    THFloatTensor_fill(grad_input, 1);
    return _TORCH_SUCCESS;
}
