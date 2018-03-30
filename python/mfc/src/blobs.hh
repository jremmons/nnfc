#pragma once

#include "blob1d.hh"
#include "blob3d.hh"
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
