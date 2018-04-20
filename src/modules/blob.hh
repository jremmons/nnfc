#pragma once

#include <Eigen/CXX11/Tensor>

#include <cassert>
#include <cstring>
#include <cstdarg>
#include <memory>

template <typename T, size_t ndims>
class Blob {
protected:
    const size_t ndims_;
    Eigen::DSizes<Eigen::Index, ndims> size_;
    T* data_; // we do not own this data

    void set_tensor(void) {
        Eigen::TensorMap<Eigen::Tensor<T, ndims>> new_tensor{data_, size_};
        std::memcpy(&tensor, &new_tensor, sizeof(Eigen::TensorMap<Eigen::Tensor<T, ndims>>));
        //tensor = new_tensor;
    }
    
public:

    Eigen::TensorMap<Eigen::Tensor<T, ndims>> tensor;

    Blob(T* data, ...) :
        ndims_(ndims),
        data_(data),
        tensor(Eigen::TensorMap<Eigen::Tensor<T, ndims>>(nullptr, size_))
    {
        va_list args;
        va_start(args, data);

        for (size_t i = 0; i < ndims; i++) {
            size_[i] = va_arg(args, size_t);
        }

        va_end(args);

        set_tensor();
        
    }
    Blob(const Blob<T, ndims>&) = delete;
    ~Blob() {}

    Blob<T, ndims>& operator=(Blob<T, ndims> &rhs) = delete;    
    
    size_t dimension(size_t dim) const {
        return size_[dim];
    }

    size_t size(void) const {
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
