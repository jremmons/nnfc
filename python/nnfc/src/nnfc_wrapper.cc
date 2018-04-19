#include <TH/TH.h>
#include <stdlib.h>

#include <cassert>
#include <cstdint>
#include <iostream>

#include<Eigen/Dense>
#include<Eigen/CXX11/Tensor>

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
        ASSERT(input_contiguous);
        int input_ndims = THFloatTensor_nDimension(input);
        ASSERT(input_ndims == 4);
    }

    // munge the blobs
    size_t n_size = THFloatTensor_size(input, 0);
    size_t c_size = THFloatTensor_size(input, 1);
    size_t h_size = THFloatTensor_size(input, 2);
    size_t w_size = THFloatTensor_size(input, 3);
    float* input_data = THFloatTensor_data(input);
    Blob4DTorchFloat input_blob{input_data, n_size, c_size, h_size, w_size};

    // Eigen::TensorMap<Eigen::Tensor<float, 4>> input_eigen_blob{input_data, n_size, c_size, h_size, w_size};
    // std::cerr << "the (0,0,0,0) value: " << input_eigen_blob(0,0,0,0) << std::endl;    

    size_t b_size = THByteTensor_nElement(output);
    uint8_t *output_data = THByteTensor_data(output);
    Blob1DTorchByte output_blob{output_data, b_size, output};

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
        ASSERT(input_contiguous);
        
        int input_ndims = THByteTensor_nDimension(input);
        ASSERT(input_ndims == 1);
    }
    
    // munge the blobs
    size_t b_size = THByteTensor_nElement(input);
    uint8_t* input_data = THByteTensor_data(input);
    Blob1DTorchByte input_blob{input_data, b_size};
    
    size_t n_size = THFloatTensor_nElement(output);
    float* output_data = THFloatTensor_data(output);
    Blob4DTorchFloat output_blob{output_data, n_size, 0, 0, 0, output};

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
