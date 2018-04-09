#pragma once

#include <cassert>
#include <cstring>
#include <memory>
#include <iostream>

template <typename T>
class Blob1D {
public:

    const size_t &size;
    T* &data;

    Blob1D(T* _data, size_t _size) :
        size(size_),
        data(data_),
        size_(_size),
        data_(_data) 
    { } 
    Blob1D(const Blob1D<T>&) = delete;
    ~Blob1D() {}

    Blob1D<T>& operator=(Blob1D<T> &rhs) = delete;

    virtual void resize(size_t) { throw; } 

    inline T get(const size_t i) const {

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
