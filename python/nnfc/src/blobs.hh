#pragma once

#include <cassert>
#include <cstdarg>

#include "blob.hh"
#include "blob1d.hh"
#include "blob4d.hh"
#include "common.hh"

class TorchFloatBlob4D : public Blob<float, 4> {
private:
    THFloatTensor *tensor_;

public:
    TorchFloatBlob4D(THFloatTensor *tensor, float *data, size_t n, size_t c, size_t h, size_t w) :
        Blob<float, 4>(data, n, c, h, w),
        tensor_(tensor)
    { }
    
    void resize(size_t n, ...) {

        va_list args;
        va_start(args, n);
        size_t c = va_arg(args, size_t);
        size_t h = va_arg(args, size_t);
        size_t w = va_arg(args, size_t);
        va_end(args);
        
        WrapperAssert(tensor_ != NULL, "Cannot resize tensor/blob! The PyTorch tensor is null!");

        THFloatTensor_resize4d(tensor_, n, c, h, w); // will realloc if size changes
        data_ = THFloatTensor_data(tensor_);

        int ndims = THFloatTensor_nDimension(tensor_);
        WrapperAssert(ndims == ndims_, "resize failed! the number of dimension changed");
        
        size_[0] = THFloatTensor_size(tensor_, 0);
        size_[1] = THFloatTensor_size(tensor_, 1);
        size_[2] = THFloatTensor_size(tensor_, 2);
        size_[3] = THFloatTensor_size(tensor_, 3);
        
        WrapperAssert(size_[0] == n, "resize failed! 'n' was not set.");
        WrapperAssert(size_[1] == c, "resize failed! 'c' was not set.");
        WrapperAssert(size_[2] == h, "resize failed! 'h' was not set.");
        WrapperAssert(size_[3] == w, "resize failed! 'w' was not set.");

        set_tensor();

    }

};

class TorchByteBlob1D : public Blob<uint8_t, 1> {
private:
    THByteTensor *tensor_;

public:
    TorchByteBlob1D(THByteTensor *tensor, uint8_t *data, size_t n) :
        Blob<uint8_t, 1>(data, n),
        tensor_(tensor)
    { }
    
    void resize(size_t n, ...) {

        WrapperAssert(tensor_ != NULL, "Cannot resize tensor/blob! The PyTorch tensor is null!");

        THByteTensor_resize1d(tensor_, n); // will realloc if size changes
        data_ = THByteTensor_data(tensor_);

        int ndims = THByteTensor_nDimension(tensor_);
        WrapperAssert(ndims == ndims_, "resize failed! the number of dimension changed");
        
        size_[0] = THByteTensor_size(tensor_, 0);
        
        WrapperAssert(size_[0] == n, "resize failed! 'n' was not set.");

        set_tensor();

    }

};































class Blob1DTorchByte : public Blob1D<uint8_t> {
public:
    Blob1DTorchByte(uint8_t *data, size_t size, THByteTensor *tensor=NULL) :
        Blob1D<uint8_t>(data, size),
        tensor_(tensor)
    { }
    
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
        
        ASSERT(size_ == new_size);
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
        
        ASSERT(batch_size_ == new_n);
        ASSERT(channels_ == new_c);
        ASSERT(height_ == new_h);
        ASSERT(width_ == new_w);
        ASSERT(size_ == new_n * new_c * new_h * new_w);
    }

private:
    THFloatTensor *tensor_;
};
