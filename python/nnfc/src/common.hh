#ifndef _NNFC_COMMON
#define _NNFC_COMMON

extern "C" {
#include <Python.h>
#include <numpy/arrayobject.h>
}

#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "tensor.hh"

inline void WrapperAssert(bool expr, const std::string message="unnammed runtime error"){
    #ifdef _NNFC_DEBUG 
    if(!expr){
        throw std::runtime_error(message);
    }
    #endif
}

inline std::vector<NNFC::Tensor<float, 3>> array2tensor(PyArrayObject *input_array) {
    
    if(!PyArray_Check(input_array)) {
        PyErr_SetString(PyExc_ValueError, "the input to the encoder must be a 4D numpy.ndarray.");
    }

    if(!PyArray_ISCARRAY(input_array)){
        PyErr_SetString(PyExc_ValueError, "the input array must be a c-style array and conriguous in memory.");
    }

    if(PyArray_TYPE(input_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_ValueError, "the input array must have dtype float32.");
    }

    const int ndims = PyArray_NDIM(input_array);
    if(ndims != 4) {
        std::string error_message("the input to the encoder must be a 4D numpy.ndarray. (The input dimensionality was: ");
        error_message = error_message + std::to_string(ndims) + std::string(")");
        PyErr_SetString(PyExc_ValueError,  error_message.c_str());
    }

    Eigen::Index dim0 = PyArray_DIM(input_array, 0);
    Eigen::Index dim1 = PyArray_DIM(input_array, 1);
    Eigen::Index dim2 = PyArray_DIM(input_array, 2);
    Eigen::Index dim3 = PyArray_DIM(input_array, 3);

    size_t stride0 = PyArray_STRIDE(input_array, 0);
    size_t nElements = PyArray_SIZE(input_array);
        
    float *data = static_cast<float*>(PyArray_DATA(input_array));

    std::vector<NNFC::Tensor<float, 3>> tensors;
    tensors.reserve(dim0);

    for(size_t offset = 0; offset < nElements; offset += stride0) {
        tensors.emplace_back(data + offset, dim1, dim2, dim3);
    }

    return tensors;
}

#endif // _NNFC_COMMON
