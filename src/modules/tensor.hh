#pragma once

#include <Eigen/CXX11/Tensor>

#include <cassert>
#include <cstring>
#include <cstdarg>
#include <memory>
#include <string>

namespace NNFC {

    template <typename T, int ndims>
    class Tensor {
    private:
        std::unique_ptr<T, void(*)(T*)> data_;
        Eigen::TensorMap<Eigen::Tensor<T, ndims>> tensor_;

    public:
        template<typename... DimSizes>
        Tensor(T* data, DimSizes&&... dims) :
            data_(data, [](T*){}), 
            tensor_(Eigen::TensorMap<Eigen::Tensor<T, ndims>>(data_.get(), Eigen::DSizes<Eigen::Index, ndims>(dims...)))
        {            
            // note: the deleter for the data_ member was set to a noop
            // because when you use this constructor data_ is not owned by
            // this object. 
        }
        
        // TODO(jremmons) get rid of the "new" for memory allocation.
        // Should be replaced by something like std::make_unique.
        template<typename... DimSizes>
        Tensor(DimSizes&&... dims) :
            data_(new T[Eigen::DSizes<Eigen::Index, ndims>(dims...).TotalSize()], [](T* ptr){ delete ptr; }), 
            tensor_(Eigen::TensorMap<Eigen::Tensor<T, ndims>>(data_.get(), Eigen::DSizes<Eigen::Index, ndims>(dims...)))
        { }

        Tensor(const Tensor<T, ndims>&) = delete;
        Tensor(const Tensor<T, ndims>&&) = delete;
        ~Tensor() { }

        Tensor<T, ndims>& operator=(Tensor<T, ndims> &rhs) = delete;    

        Eigen::Index dimension(Eigen::Index dim) const
        {
            return tensor_.dimension(dim);
        }

        Eigen::Index size(void) const
        {
            return tensor_.size();
        }

        Eigen::Index rank(void) const
        {
            return tensor_.rank();
        }

        template<typename... Indices>
        T operator()(Indices&& ...indices)
        {
            return tensor_(indices...);
        }
        
    };

}
