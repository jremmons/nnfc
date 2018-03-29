#include <TH/TH.h>
#include <stdlib.h>

#include <cassert>
#include <cstdint>
#include <iostream>

#include "noop.hh"
#include "blob1d.hh"
#include "blob3d.hh"

// functions must 'extern "C"' in order to be callable from within pytorch/python
// https://github.com/torch/TH/blob/master/generic/THTensor.h

#define _TORCH_SUCCESS 1;
#define _TORCH_FAILURE 0;

#define DEBUG 1
#define ASSERT(expr) if(!(expr) and DEBUG){ std::cerr << "assertion failed at line: " <<  __LINE__ << " \n"; throw; } // TOOD(jremmons) make this more sensible...



class Blob1DTorchByte : public Blob1D<uint8_t> {
public:
    Blob1DTorchByte(uint8_t *data, size_t size, THByteTensor *tensor) :
        Blob1D<uint8_t>(data, size),
        tensor_(tensor)
    { }
    
    void resize(size_t new_size) {
        THByteTensor_resize1d(tensor_, new_size); // will realloc
        data_ = THByteTensor_data(tensor_);
        size_ = THByteTensor_size(tensor_, 0);
        
        ASSERT(size_ == new_size);
    }

private:
    THByteTensor *tensor_;
};


class Blob3DTorchFloat : public Blob3D<float> {
public:
    Blob3DTorchFloat(float *data, size_t c, size_t h, size_t w, THFloatTensor *tensor) :
        Blob3D<float>(data, c, h, w),
        tensor_(tensor)
    { }
    
    void resize(size_t new_c, size_t new_h, size_t new_w) {
        THFloatTensor_resize3d(tensor_, new_c, new_h, new_w); // will realloc
        data_ = THFloatTensor_data(tensor_);

        channels_ = THFloatTensor_size(tensor_, 0);
        height_ = THFloatTensor_size(tensor_, 1);
        width_ = THFloatTensor_size(tensor_, 2);

        size_ = channels_ * height_ * width_;

        channels_stride_ = height_ * width_;
        height_stride_ = width_;
        width_stride_ = 1;
        
        ASSERT(channels_ == new_c);
        ASSERT(width_ == new_h);
        ASSERT(height_ == new_w);
        ASSERT(size_ == new_c * new_h * new_w);
    }

private:
    THFloatTensor *tensor_;
};


extern "C" int encode_forward(THFloatTensor *input, THByteTensor *output)
{

    // sanity checking
    {
        int input_contiguous = THFloatTensor_isContiguous(input);
        ASSERT(input_contiguous);
        int input_ndims = THFloatTensor_nDimension(input);
        ASSERT(input_ndims == 3);
    }
    
    // munge the blobs
    size_t c_size = THFloatTensor_size(input, 0);
    size_t h_size = THFloatTensor_size(input, 1);
    size_t w_size = THFloatTensor_size(input, 2);
    float* input_data = THFloatTensor_data(input);
    Blob3DTorchFloat input_blob{input_data, c_size, h_size, w_size, input};

    size_t b_size = THByteTensor_nElement(output);
    uint8_t *output_data = THByteTensor_data(output);
    Blob1DTorchByte output_blob{output_data, b_size, output};

    // call the encoder
    Noop n;
    n.encode(input_blob, output_blob);

    return _TORCH_SUCCESS;    
}


extern "C" int encode_backward(THFloatTensor *grad_output, THFloatTensor *grad_input)
{
    THFloatTensor_resizeAs(grad_input, grad_output);
    THFloatTensor_fill(grad_input, 1);
    return _TORCH_SUCCESS;
}


extern "C" int decode_forward(THByteTensor *input, THFloatTensor *output)
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
    Blob1DTorchByte input_blob{input_data, b_size, input};
    
    size_t c_size = THFloatTensor_nElement(output);
    float* output_data = THFloatTensor_data(output);
    Blob3DTorchFloat output_blob{output_data, c_size, 1, 1, output};

    // call the decoder
    Noop n;
    n.decode(input_blob, output_blob);

    return _TORCH_SUCCESS;    
}


extern "C" int decode_backward(THFloatTensor *grad_output, THFloatTensor *grad_input)
{
    THFloatTensor_resizeAs(grad_input, grad_output);
    THFloatTensor_fill(grad_input, 1);
    return _TORCH_SUCCESS;
}
