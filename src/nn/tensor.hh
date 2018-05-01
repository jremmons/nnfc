#ifndef _NN_TENSOR_H
#define _NN_TENSOR_H

#include <Eigen/CXX11/Tensor>

#include <cassert>
#include <cstring>
#include <cstdarg>
#include <memory>
#include <string>

namespace nn {

    typedef Eigen::Index Index;

    template<typename T>
    class Rawptr {
    private:
        std::shared_ptr<T> ptr_;
        size_t size_;

    public:
        rawptr(T* ptr, size_t size) :
            ptr_(ptr),
            size_(size)
        { }

        ~rawptr() { }        

        std::shared_ptr<T> get() { return ptr_; }
        size_t size() { return size_; }
    };
    
    template <typename T, int ndims>
    class Tensor {
    private:
        Eigen::DSizes<Eigen::Index, ndims> size_;
        std::shared_ptr<T> data_;
        Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> tensor_;

        T* data()
        {
            return tensor_.data();
        }
        
    public:
        template<typename... DimSizes>
        Tensor(T* data, DimSizes&&... dims) :
            size_(dims...),
            data_(data, [](T*){}), 
            tensor_(Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>>(data_.get(), size_))
        {            
            // note: the deleter for the data_ member was set to a noop
            // because when you use this constructor data_ is not owned
            // by this object. 
        }

        // TODO(jremmons) get rid of the "new" for memory allocation.
        // Should be replaced by something like std::make_unique.
        template<typename... DimSizes>
        Tensor(DimSizes... dims) :
            size_(dims...),
            data_(new T[size_.TotalSize()], [](T* ptr){ delete ptr; }), 
            tensor_(Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>>(data_.get(), size_))
        { }

        // make this the real constructor for the object
        Tensor(T* data, Eigen::DSizes<Eigen::Index, ndims> size) :
            size_(size),
            data_(data, [](T*){}),
            tensor_(Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>>(data_.get(), size_))
        { }
               
        Tensor(Eigen::DSizes<Eigen::Index, ndims> size) :
            size_(size),
            data_(new T[size_.TotalSize()]),
            tensor_(Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>>(data_.get(), size_))
        { }
        
        Tensor(Tensor<T, ndims>&& other) noexcept :
            size_(std::move(other.size_)),
            data_(std::move(other.data_)),
            tensor_(Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>>(data_.get(), size_))
        { }

        Tensor(const Tensor<T, ndims>& other) noexcept :
            size_(other.size_),
            data_(other.data_),
            tensor_(Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>>(data_.get(), size_))
        { }

        ~Tensor() { }

        Tensor<T, ndims> operator=(Tensor<T, ndims> rhs) = delete;

        Tensor deepcopy() const
        {
            Tensor<T, ndims> new_tensor(size_);
            std::memcpy(new_tensor.data(), tensor_.data(), size_.TotalSize());
            return new_tensor;
        }
        
        Eigen::Index dimension(Eigen::Index dim) const
        {
            return tensor_.dimension(dim);
        }

        Eigen::Index size() const
        {
            return tensor_.size();
        }

        Eigen::Index rank() const
        {
            return tensor_.rank();
        }

        template<typename... Indices>
        inline T& operator()(Indices&& ...indices)
        {
            return tensor_(indices...);
        }

        template<typename... Indices>
        inline const T& operator()(Indices&& ...indices) const
        {
            return tensor_(indices...);
        }

    };

}

#endif // _NN_TENSOR_H
