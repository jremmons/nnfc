#pragma once

#include <cassert>
#include <cstring>
#include <memory>

template <typename T>
class Blob3D {
public:

    Blob3D(T* data, const size_t c, const size_t h, const size_t w) :
        channels_(c),
        height_(h),
        width_(w),
        size_(c*h*w),
        channels_stride_(h*w),
        height_stride_(w),
        width_stride_(1),
        data_(data)
    { }    
    ~Blob3D() {}

    inline T get(const size_t ci, const size_t hi, const size_t wi){

        assert(0 <= ci and ci < channels_);
        assert(0 <= hi and hi < height_);
        assert(0 <= wi and wi < width_);
        
        size_t offset = channels_stride_ * ci + height_stride_ * hi + width_stride_ * wi;
        assert(0 <= offset and offset < size_);
        
        return data_[offset];
    }

    inline void set(const T value, const size_t ci, const size_t hi, const size_t wi){

        assert(0 <= ci and ci < channels_);
        assert(0 <= hi and hi < height_);
        assert(0 <= wi and wi < width_);
        
        size_t offset = channels_stride_ * ci + height_stride_ * hi + width_stride_ * wi;
        assert(0 <= offset and offset < size_);

        data_[offset] = value;
    }
    
private:
    const size_t channels_;
    const size_t height_;
    const size_t width_;
    const size_t size_;

    const size_t channels_stride_;
    const size_t height_stride_;
    const size_t width_stride_;

    T* data_; // we don't allocate this data
};
