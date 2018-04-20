#pragma once

#include <cassert>
#include <cstdarg>

#include "blob.hh"
#include "blob1d.hh"
#include "blob4d.hh"
#include "common.hh"

class TorchFloatBlob4D : public Blob<float, 4> {
private:
    THFloatTensor *THtensor_;

public:

    float *&data;
    
    TorchFloatBlob4D(THFloatTensor *THtensor, float *data, Eigen::Index n, Eigen::Index c, Eigen::Index h, Eigen::Index w) :
        Blob<float, 4>(data, n, c, h, w),
        THtensor_(THtensor),
        data(data_)
    { }
    TorchFloatBlob4D(const TorchFloatBlob4D&) = delete;
    TorchFloatBlob4D(const TorchFloatBlob4D&&) = delete; 
    TorchFloatBlob4D& operator=(TorchFloatBlob4D &rhs) = delete;    
   
    void resize(Eigen::Index n, ...) {

        va_list args;
        va_start(args, n);
        Eigen::Index c = va_arg(args, Eigen::Index);
        Eigen::Index h = va_arg(args, Eigen::Index);
        Eigen::Index w = va_arg(args, Eigen::Index);
        va_end(args);
        
        WrapperAssert(THtensor_ != NULL, "Cannot resize tensor/blob! The PyTorch tensor is null!");

        THFloatTensor_resize4d(THtensor_, n, c, h, w); // will realloc if size changes
        data_ = THFloatTensor_data(THtensor_);

        int ndims = THFloatTensor_nDimension(THtensor_);
        WrapperAssert(ndims == ndims_, "resize failed! the number of dimension changed");
        
        size_[0] = THFloatTensor_size(THtensor_, 0);
        size_[1] = THFloatTensor_size(THtensor_, 1);
        size_[2] = THFloatTensor_size(THtensor_, 2);
        size_[3] = THFloatTensor_size(THtensor_, 3);
        
        WrapperAssert(size_[0] == n, "resize failed! 'n' was not set.");
        WrapperAssert(size_[1] == c, "resize failed! 'c' was not set.");
        WrapperAssert(size_[2] == h, "resize failed! 'h' was not set.");
        WrapperAssert(size_[3] == w, "resize failed! 'w' was not set.");

        set_tensor();

    }

};

class TorchByteBlob1D : public Blob<uint8_t, 1> {
private:
    THByteTensor *THtensor_;

public:

    uint8_t *&data;
    
    TorchByteBlob1D(THByteTensor *THtensor, uint8_t *data, Eigen::Index n) :
        Blob<uint8_t, 1>(data, n),
        THtensor_(THtensor),
        data(data_)
    { }
    TorchByteBlob1D(const TorchByteBlob1D&) = delete;
    TorchByteBlob1D(const TorchByteBlob1D&&) = delete;
    TorchByteBlob1D& operator=(TorchByteBlob1D &rhs) = delete;
    
    void resize(Eigen::Index n, ...) {

        WrapperAssert(THtensor_ != NULL, "Cannot resize tensor/blob! The PyTorch tensor is null!");

        THByteTensor_resize1d(THtensor_, n); // will realloc if size changes
        data_ = THByteTensor_data(THtensor_);

        int ndims = THByteTensor_nDimension(THtensor_);
        WrapperAssert(ndims == ndims_, "resize failed! the number of dimension changed");
        
        size_[0] = THByteTensor_size(THtensor_, 0);
        
        WrapperAssert(size_[0] == n, "resize failed! 'n' was not set.");

        set_tensor();

    }

};
