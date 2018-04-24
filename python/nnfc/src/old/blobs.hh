#pragma once

#include <cassert>
#include <cstdarg>

#include "blob.hh"
#include "common.hh"

class TorchFloatBlob4D : public Blob<float, 4> {
private:
    THFloatTensor *THtensor_;

    inline size_t THTensor_size_safe(THFloatTensor *THtensor, int dim) {
        int size = THFloatTensor_nElement(THtensor);
        if(size <= 0){
            return 0;
        }

        int nDims = THFloatTensor_nDimension(THtensor);
        if(dim >= nDims){
            return 0;
        }

        return THFloatTensor_size(THtensor, dim);
    }
    
public:

    float *&data;
    
    TorchFloatBlob4D(THFloatTensor *THtensor, float *data, Eigen::Index n, Eigen::Index c, Eigen::Index h, Eigen::Index w) :
        Blob<float, 4>(data, n, c, h, w),
        THtensor_(THtensor),
        data(data_)
    { }
    TorchFloatBlob4D(THFloatTensor *THtensor) :
        Blob<float, 4>(THFloatTensor_data(THtensor),
                       THTensor_size_safe(THtensor, 0),
                       THTensor_size_safe(THtensor, 1),
                       THTensor_size_safe(THtensor, 2),
                       THTensor_size_safe(THtensor, 3)),
        THtensor_(THtensor),
        data(data_)
    {
        size_t size = THFloatTensor_nElement(THtensor);
        size_t dim0_size = THTensor_size_safe(THtensor, 0);
        size_t dim1_size = THTensor_size_safe(THtensor, 1);
        size_t dim2_size = THTensor_size_safe(THtensor, 2);
        size_t dim3_size = THTensor_size_safe(THtensor, 3);
        WrapperAssert(size == dim0_size * dim1_size * dim2_size * dim3_size, "size does not match dims");

        if(size > 0) {
            int input_contiguous = THFloatTensor_isContiguous(THtensor);
            WrapperAssert(input_contiguous, "input tensor not contiguous!");

            int input_ndims = THFloatTensor_nDimension(THtensor);
            WrapperAssert(input_ndims == 4, "input dimensions must be 4");

        }
        
    }
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

        size_[0] = THTensor_size_safe(THtensor_, 0);
        size_[1] = THTensor_size_safe(THtensor_, 1);
        size_[2] = THTensor_size_safe(THtensor_, 2);
        size_[3] = THTensor_size_safe(THtensor_, 3);
        
        WrapperAssert(size_[0] == n, "resize failed! 'n' was not set.");
        WrapperAssert(size_[1] == c, "resize failed! 'c' was not set.");
        WrapperAssert(size_[2] == h, "resize failed! 'h' was not set.");
        WrapperAssert(size_[3] == w, "resize failed! 'w' was not set.");

        int ndims = THFloatTensor_nDimension(THtensor_);
        WrapperAssert(ndims == ndims_, "resize failed! the number of dimension changed");
        
        int input_contiguous = THFloatTensor_isContiguous(THtensor_);
        WrapperAssert(input_contiguous, "resize failed! input tensor not contiguous!");

        set_tensor();
        
    }

};


class TorchByteBlob1D : public Blob<uint8_t, 1> {
private:
    THByteTensor *THtensor_;

    inline size_t THTensor_size_safe(THByteTensor *THtensor, int dim) {
        int size = THByteTensor_nElement(THtensor);
        if(size <= 0){
            return 0;
        }

        int nDims = THByteTensor_nDimension(THtensor);
        if(dim >= nDims){
            return 0;
        }

        return THByteTensor_size(THtensor, dim);
    }
    
public:

    uint8_t *&data;
    
    TorchByteBlob1D(THByteTensor *THtensor, uint8_t *data, Eigen::Index n) :
        Blob<uint8_t, 1>(data, n),
        THtensor_(THtensor),
        data(data_)
    { }
    TorchByteBlob1D(THByteTensor *THtensor) :
        Blob<uint8_t, 1>(THByteTensor_data(THtensor),
                         THTensor_size_safe(THtensor, 0)),
        THtensor_(THtensor),
        data(data_)
    {
        size_t size = THByteTensor_nElement(THtensor);
        size_t dim0_size = THTensor_size_safe(THtensor, 0);
        WrapperAssert(size == dim0_size, "size does not match dims");

        if(size > 0) {
            int input_contiguous = THByteTensor_isContiguous(THtensor);
            WrapperAssert(input_contiguous, "input tensor not contiguous!");

            int input_ndims = THByteTensor_nDimension(THtensor);
            WrapperAssert(input_ndims == 1, "input dimensions must be 1");
        }
        
    }
    TorchByteBlob1D(const TorchByteBlob1D&) = delete;
    TorchByteBlob1D(const TorchByteBlob1D&&) = delete;
    TorchByteBlob1D& operator=(TorchByteBlob1D &rhs) = delete;
    
    void resize(Eigen::Index n, ...) {

        WrapperAssert(THtensor_ != NULL, "Cannot resize tensor/blob! The PyTorch tensor is null!");

        THByteTensor_resize1d(THtensor_, n); // will realloc if size changes
        data_ = THByteTensor_data(THtensor_);

        size_[0] = THTensor_size_safe(THtensor_, 0);
        
        WrapperAssert(size_[0] == n, "resize failed! 'n' was not set.");

        int ndims = THByteTensor_nDimension(THtensor_);
        WrapperAssert(ndims == ndims_, "resize failed! the number of dimension changed");
        
        int input_contiguous = THByteTensor_isContiguous(THtensor_);
        WrapperAssert(input_contiguous, "resize failed! input tensor not contiguous!");

        set_tensor();
            
    }

};
