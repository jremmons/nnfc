#pragma once

#include <cassert>
#include <cstring>
#include <memory>
#include <iostream>

template <typename T>
class Blob1D {
public:

    Blob1D(T* data, size_t size) :
        size_(size),
        data_(data)
    { }
    ~Blob1D() {}

    virtual void resize(size_t) = 0; 

    inline T get(const size_t i){

        assert(0 <= i and i < size_);
        return data_[i];
    }

    inline void set(const T value, const size_t i){

        assert(0 <= i and i < size_);
        data_[i] = value;
    }
    
protected:
    size_t size_;
    T* data_; // we don't allocate this data
};
