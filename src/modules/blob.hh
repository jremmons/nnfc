#pragma once

#include <Eigen/CXX11/Tensor>

#include <cassert>
#include <cstring>
#include <cstdarg>
#include <memory>

template <typename T, int ndims>
class Blob {
protected:
    const int ndims_;
    Eigen::DSizes<Eigen::Index, ndims> size_;
    T* data_; // we do not own this data

    void set_tensor(void) {
        Eigen::TensorMap<Eigen::Tensor<T, ndims>> new_tensor{data_, size_};
        std::memcpy(&tensor, &new_tensor, sizeof(Eigen::TensorMap<Eigen::Tensor<T, ndims>>));
    }
    
public:

    Eigen::TensorMap<Eigen::Tensor<T, ndims>> tensor;

    Blob(T* data, ...) :
        ndims_(ndims),
        size_(),
        data_(data),
        tensor(Eigen::TensorMap<Eigen::Tensor<T, ndims>>(nullptr, size_))
    {
        va_list args;
        va_start(args, data);

        for (int i = 0; i < ndims; i++) {
            size_[i] = va_arg(args, Eigen::Index);
        }

        va_end(args);
        set_tensor();
        
    }
    Blob(Eigen::TensorMap<Eigen::Tensor<T, ndims>> t) :
        ndims_(ndims),
        size_(),
        data_(nullptr),
        tensor(Eigen::TensorMap<Eigen::Tensor<T, ndims>>(nullptr, size_))
    {
        throw std::runtime_error("not yet implemented");
    }
    Blob(const Blob<T, ndims>&) = delete;
    Blob(const Blob<T, ndims>&&) = delete;
    virtual ~Blob() {}
    
    Blob<T, ndims>& operator=(Blob<T, ndims> &rhs) = delete;    
    
    Eigen::Index dimension(Eigen::Index dim) const {
        return size_[dim];
    }

    Eigen::Index size(void) const {
        Eigen::Index s = 1;

        for(int i = 0; i < ndims; i++){
            s *= dimension(i);
        }

        return s;
    }

    Eigen::TensorMap<Eigen::Tensor<T, ndims>> get_tensor(void) const {
        Eigen::DSizes<long int, ndims> dimensions;

        for(int i = 0; i < ndims; i++){
            dimensions[i] = size_[i];
        }

        Eigen::TensorMap<Eigen::Tensor<T, ndims>> tensor{data_, dimensions};
        return tensor;
    }
    
    virtual void resize(Eigen::Index, ...) {
        throw std::runtime_error("resize is not defined in the base class of 'blob'");
    }    
};
