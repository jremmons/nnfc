#pragma once

#include <cassert>
#include <cstring>
#include <memory>

template <typename T>
class Blob4D {
public:

    const size_t &batch_size;
    const size_t &channels;

    const size_t &out_channels;
    const size_t &in_channels;    
    
    const size_t &height;
    const size_t &width;
    const size_t &size;

    T* &data;
    
    Blob4D(T* data, size_t n, size_t c, size_t h, size_t w) :
        batch_size(batch_size_),
        channels(channels_),
        out_channels(batch_size_),
        in_channels(channels_),
        height(height_),
        width(width_),
        size(size_),
        data(data_),
        batch_size_(n),
        channels_(c),
        height_(h),
        width_(w),
        size_(n*c*h*w),
        batch_stride_(c*h*w),
        channels_stride_(h*w),
        height_stride_(w),
        width_stride_(1),
        data_(data)
    { }    
    Blob4D(const Blob4D<T>&) = delete;
    Blob4D(const Blob4D<T>&&) = delete;
    virtual ~Blob4D() {}

    Blob4D<T>& operator=(Blob4D<T> &rhs) = delete;
    
    virtual void resize(size_t, size_t, size_t, size_t) { throw; }
    
    inline T get(const size_t ni, const size_t ci, const size_t hi, const size_t wi) const {

        assert(0 <= ni and ni < batch_size_);
        assert(0 <= ci and ci < channels_);
        assert(0 <= hi and hi < height_);
        assert(0 <= wi and wi < width_);
        
        size_t offset = batch_stride_ * ni + channels_stride_ * ci + height_stride_ * hi + width_stride_ * wi;
        assert(0 <= offset and offset < size_);
        
        return data_[offset];
    }

    inline void set(const T value, const size_t ni, const size_t ci, const size_t hi, const size_t wi){

        assert(0 <= ni and ni < batch_size_);
        assert(0 <= ci and ci < channels_);
        assert(0 <= hi and hi < height_);
        assert(0 <= wi and wi < width_);
        
        size_t offset = batch_stride_ * ni + channels_stride_ * ci + height_stride_ * hi + width_stride_ * wi;
        assert(0 <= offset and offset < size_);

        data_[offset] = value;
    }
    
protected:
    size_t batch_size_;
    size_t channels_;
    size_t height_;
    size_t width_;
    size_t size_;

    size_t batch_stride_;
    size_t channels_stride_;
    size_t height_stride_;
    size_t width_stride_;

    T* data_; // we don't allocate this data
};
