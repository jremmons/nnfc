#pragma once

#include <Eigen/Dense>
#include <Eigen/CXX11/Tensor>

#include <cassert>
#include <cstring>
#include <cstdarg>
#include <memory>

template <typename T, size_t ndims>
class Blob {
protected:
    const size_t ndims_;
    size_t size_[ndims];
    T* data_; // we do not own this data

    
public:
    Blob(T* data, ...) :
        ndims_(ndims),
        data_(data)
    {
        va_list args;
        va_start(args, data);

        for (size_t i = 0; i < ndims; i++) {
            size_[i] = va_arg(args, size_t);
        }

        va_end(args);
        
    }
    Blob(const Blob<T, ndims>&) = delete;
    ~Blob() {}


    Blob<T, ndims>& operator=(Blob<T, ndims> &rhs) = delete;    

    
    size_t dimension(size_t dim) {
        return size_[dim];
    }

    
    size_t size(void) {
        size_t s = 1;

        for(size_t i = 0; i < ndims; i++){
            s *= dimension(i);
        }

        return s;
    }

    
    Eigen::TensorMap<Eigen::Tensor<T, ndims>> get_tensor(void) const {
        Eigen::DSizes<long int, ndims> dimensions;

        for(size_t i = 0; i < ndims; i++){
            dimensions[i] = size_[i];
        }
        
        Eigen::TensorMap<Eigen::Tensor<T, ndims>> tensor{data_, dimensions};
        return tensor;
    }

    
    virtual void resize(size_t, ...) {
        throw std::runtime_error("resize is not defined in the base class of 'blob'");
    }    
};
