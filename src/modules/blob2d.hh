#pragma once

#include <cassert>
#include <cstring>
#include <memory>

template <typename T>
class Blob2D {
public:
    
    const size_t &out_channels;
    const size_t &in_channels;
    const size_t &size;

    T* &data;

    Blob2D(T* data, size_t _out_channels, size_t _in_channels) :
        out_channels(out_channels_),
        in_channels(in_channels_),
        size(size_),
        data(data_),
        out_channels_(_out_channels),
        in_channels_(_in_channels),
        size_(_out_channels*_in_channels),
        out_channels_stride_(_in_channels),
        in_channels_stride_(1),
        data_(data)
    { }    
    Blob2D(const Blob2D<T>&) = delete;
    ~Blob2D() {}

    Blob2D<T>& operator=(Blob2D<T> &rhs) = delete;

    virtual void resize(size_t, size_t) { throw std::runtime_error("resize is not implemented in the base class"); }
    
    inline T get(const size_t oi, const size_t ii) const {

        assert(0 <= oi and oi < out_channels_);
        assert(0 <= ii and ii < in_channels_);
        
        size_t offset = out_channels_stride_ * oi + in_channels_stride_ * ii;
        assert(0 <= offset and offset < size_);

        return data_[offset];
    }

    inline void set(const T value, const size_t oi, const size_t ii){

        assert(0 <= oi and oi < out_channels_);
        assert(0 <= ii and ii < in_channels_);
        
        size_t offset = out_channels_stride_ * oi + in_channels_stride_ * ii;
        assert(0 <= offset and offset < size_);

        data_[offset] = value;
    }
    
protected:
    size_t out_channels_;
    size_t in_channels_;
    size_t size_;

    size_t out_channels_stride_;
    size_t in_channels_stride_;

    T* data_; // we don't allocate this data
};
