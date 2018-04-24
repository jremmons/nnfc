#include <TH/TH.h>
#include <stdlib.h>

#include <cassert>
#include <cstdint>
#include <iostream>

#include "blob1d.hh"
#include "blob4d.hh"
#include "common.hh"
#include "noop.hh"

// functions must 'extern "C"' in order to be callable from within pytorch/python
// https://github.com/torch/TH/blob/master/generic/THTensor.h

class Blob1DTorchByte : public Blob1D<uint8_t> {
public:
    Blob1DTorchByte(uint8_t *data, size_t size, THByteTensor *tensor=NULL) :
        Blob1D<uint8_t>(data, size),
        tensor_(tensor)
    { }
    Blob1DTorchByte(const Blob1DTorchByte&) = delete;
    Blob1DTorchByte(const Blob1DTorchByte&&) = delete;
    Blob1DTorchByte& operator=(Blob1DTorchByte &rhs) = delete;
    
    void resize(size_t new_size) {
        if(!tensor_){
            throw std::runtime_error("cannot resize tensor! The PyTorch tensor is null!");
        }

        if(new_size == size_){
            return; // no change in size
        }

        THByteTensor_resize1d(tensor_, new_size); // will realloc
        data_ = THByteTensor_data(tensor_);
        size_ = THByteTensor_size(tensor_, 0);
        
        WrapperAssert(size_ == new_size, "resize failed! 'size' was not set correctly.");
    }
    
private:
    THByteTensor *tensor_;
};

class Blob4DTorchFloat : public Blob4D<float> {
public:
    Blob4DTorchFloat(float *data, size_t n, size_t c, size_t h, size_t w, THFloatTensor *tensor=NULL) :
        Blob4D<float>(data, n, c, h, w),
        tensor_(tensor)
    { }
    Blob4DTorchFloat(const Blob4DTorchFloat&) = delete;
    Blob4DTorchFloat(const Blob4DTorchFloat&&) = delete;
    Blob4DTorchFloat& operator=(Blob4DTorchFloat &rhs) = delete;
    
    void resize(size_t new_n, size_t new_c, size_t new_h, size_t new_w) {
        if(!tensor_){
            throw std::runtime_error("cannot resize tensor! The PyTorch tensor is null!");
        }

        if(new_n == batch_size_ and
           new_c == channels_ and
           new_h == height_ and
           new_w == width_){

            return; // no change in size
        }
        
        THFloatTensor_resize4d(tensor_, new_n, new_c, new_h, new_w); // will realloc
        data_ = THFloatTensor_data(tensor_);

        batch_size_ = THFloatTensor_size(tensor_, 0);
        channels_ = THFloatTensor_size(tensor_, 1);
        height_ = THFloatTensor_size(tensor_, 2);
        width_ = THFloatTensor_size(tensor_, 3);

        size_ = batch_size_ * channels_ * height_ * width_;

        batch_stride_ = channels_ * height_ * width_;
        channels_stride_ = height_ * width_;
        height_stride_ = width_;
        width_stride_ = 1;
        
        WrapperAssert(batch_size_ == new_n, "resize failed! 'batch_size' was not set correctly.");
        WrapperAssert(channels_ == new_c, "resize failed! 'channels' was not set correctly.");
        WrapperAssert(height_ == new_h, "resize failed! 'height' was not set correctly.");
        WrapperAssert(width_ == new_w, "resize failed! 'width' was not set correctly.");
        WrapperAssert(size_ == new_n * new_c * new_h * new_w, "resize failed! 'size' was not set correctly.");
    }

private:
    THFloatTensor *tensor_;
};

extern "C" int noop_encode_forward(THFloatTensor *input, THByteTensor *output)
{
    // sanity checking
    {
        int input_contiguous = THFloatTensor_isContiguous(input);
        WrapperAssert(input_contiguous, "input array not contiguous!");
        int input_ndims = THFloatTensor_nDimension(input);
        WrapperAssert(input_ndims == 4, "input dimensions must be 4");
    }

    // munge the blobs
    size_t n_size = THFloatTensor_size(input, 0);
    size_t c_size = THFloatTensor_size(input, 1);
    size_t h_size = THFloatTensor_size(input, 2);
    size_t w_size = THFloatTensor_size(input, 3);
    float* input_data = THFloatTensor_data(input);
    Blob4DTorchFloat input_blob{input_data, n_size, c_size, h_size, w_size};

    size_t b_size = THByteTensor_nElement(output);
    uint8_t *output_data = THByteTensor_data(output);
    Blob1DTorchByte output_blob{output_data, b_size, output};

    // call the encoder
    Noop::encode(input_blob, output_blob);
    
    return _TORCH_SUCCESS;    
}


extern "C" int noop_encode_backward(THFloatTensor *grad_output, THFloatTensor *grad_input)
{
    THFloatTensor_resizeAs(grad_input, grad_output);
    THFloatTensor_fill(grad_input, 1);
    return _TORCH_SUCCESS;
}


extern "C" int noop_decode_forward(THByteTensor *input, THFloatTensor *output)
{

    // sanity checking
    {
        int input_contiguous = THByteTensor_isContiguous(input);
        WrapperAssert(input_contiguous, "input array not contiguous!");
        
        int input_ndims = THByteTensor_nDimension(input);
        WrapperAssert(input_ndims == 1, "input dimensions must be 1");
    }
    
    // munge the blobs
    size_t b_size = THByteTensor_nElement(input);
    uint8_t* input_data = THByteTensor_data(input);
    Blob1DTorchByte input_blob{input_data, b_size};
    
    size_t n_size = THFloatTensor_nElement(output);
    float* output_data = THFloatTensor_data(output);
    Blob4DTorchFloat output_blob{output_data, n_size, 0, 0, 0, output};

    // call the decoder
    Noop::decode(input_blob, output_blob);
    
    return _TORCH_SUCCESS;    
}


extern "C" int noop_decode_backward(THFloatTensor *grad_output, THFloatTensor *grad_input)
{
    THFloatTensor_resizeAs(grad_input, grad_output);
    THFloatTensor_fill(grad_input, 1);
    return _TORCH_SUCCESS;
}
