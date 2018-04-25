#pragma once

#include <Eigen/CXX11/Tensor>

#include <cassert>
#include <cstring>
#include <cstdarg>
#include <memory>

namespace NNFC {

    template <typename T, int ndims>
    class Tensor {
    private:

        Eigen::DSizes<Eigen::Index, ndims> size_;
        std::unique_ptr<T, void(*)(T*)> data_;
        Eigen::TensorMap<Eigen::Tensor<T, ndims>> tensor_;

    public:

        template<typename... DimSizes>
        Tensor(T* data, DimSizes&&... dims) :
            size_(dims...),
            data_(data, [](T*){}), 
            tensor_(Eigen::TensorMap<Eigen::Tensor<T, ndims>>(data_.get(), size_))
        {
            // note: the deleter for the data_ member was set to a noop
            // because when you use this constructor data_ is not owned by
            // this object. 
        }

        template<typename... DimSizes>
        Tensor(DimSizes&&... dims) :
            size_(dims...),
            data_(new T[size_.TotalSize()], [](T* ptr){ delete ptr; }), 
            tensor_(Eigen::TensorMap<Eigen::Tensor<T, ndims>>(data_.get(), size_))
        { }

        Tensor(const Tensor<T, ndims>&) = delete;
        Tensor(const Tensor<T, ndims>&&) = delete;
        ~Tensor() {}

        Tensor<T, ndims>& operator=(Tensor<T, ndims> &rhs) = delete;    

        Eigen::Index dimension(Eigen::Index dim) const {

            assert(size_[dim] == tensor_.dimension(dim));
            return tensor_.dimension(dim);
        }

        // Eigen::Index nElements(void) const {
        //     return ;
        // }
        
    };

}
