#pragma once

#include <cassert>
#include <cstring>
#include <memory>

template <typename T>
class Blob3D {
public:
    
    const size_t &channels;
    const size_t &height;
    const size_t &width;
    const size_t &size;

    T* &data;

    Blob3D(T* data, size_t c, size_t h, size_t w) :
        channels(channels_),
        height(height_),
        width(width_),
        size(size_),
        data(data_),
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

    virtual void resize(size_t, size_t, size_t) { throw std::runtime_error("resize is not implemented in the base class"); }
    
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
    
protected:
    size_t channels_;
    size_t height_;
    size_t width_;
    size_t size_;

    size_t channels_stride_;
    size_t height_stride_;
    size_t width_stride_;

    T* data_; // we don't allocate this data
};
