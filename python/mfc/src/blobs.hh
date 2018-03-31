#pragma once

#include "blob1d.hh"
#include "blob4d.hh"
#include "common.hh"

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

class Blob4DTorchFloat : public Blob4D<float> {
public:
    Blob4DTorchFloat(float *data, size_t n, size_t c, size_t h, size_t w, THFloatTensor *tensor) :
        Blob4D<float>(data, n, c, h, w),
        tensor_(tensor)
    { }
    
    void resize(size_t new_n, size_t new_c, size_t new_h, size_t new_w) {
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
