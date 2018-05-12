#include <Python.h>

#include <torch/torch.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include <vector>
#include <string>
#include <iostream>

#include "common.hh"
#include "nnfc_cuda.hh"

PyObject* NNFCtensor_memcpy_h2d(PyObject *, PyObject *args, PyObject *kwargs){
    // dest should be on the GPU
    // src should be on the CPU

    torch::PythonArgParser parser({ "func(Tensor dest, Tensor src)" });
    torch::ParsedArgs<2> parsed_args;
    
    auto r = parser.parse(args, kwargs, parsed_args);

    at::Tensor dest = r.tensor(0);
    at::Tensor src = r.tensor(1);

    // sanity checking
    WrapperAssert(dest.is_same_size(src), PyExc_ValueError,
                  "`dest` and `src` must be the same size.");
    
    WrapperAssert(dest.is_cuda(), PyExc_ValueError,
                  "`dest` must be a CUDA tensor.");
    WrapperAssert(!src.is_cuda(), PyExc_ValueError,
                  "`src` must be a CPU tensor.");
    
    WrapperAssert(dest.is_contiguous(), PyExc_ValueError,
                  "`dest` must be a contiguous tensor.");
    WrapperAssert(src.is_contiguous(), PyExc_ValueError,
                  "`src` must be a contiguous tensor.");
    
    WrapperAssert(dest.type().scalarType() == at::ScalarType::Float, PyExc_ValueError,
                  "`dest` must be a float32 tensor.");
    WrapperAssert(src.type().scalarType() == at::ScalarType::Float, PyExc_ValueError,
                  "`src` must be a float32 tensor.");

    // perform the memcpy
    const size_t numel = src.numel();
    void* dest_data = dest.data_ptr();
    void* src_data = src.data_ptr();

    cudaMemcpy(dest_data, src_data, numel*sizeof(float), cudaMemcpyHostToDevice);
    
    Py_RETURN_NONE;
}

PyObject* NNFCtensor_memcpy_d2h(PyObject *, PyObject *args, PyObject *kwargs){
    // dest should be on the CPU
    // src should be on the GPU

    torch::PythonArgParser parser({ "func(Tensor dest, Tensor src)" });
    torch::ParsedArgs<2> parsed_args;
    
    auto r = parser.parse(args, kwargs, parsed_args);

    at::Tensor dest = r.tensor(0);
    at::Tensor src = r.tensor(1);

    // sanity checking
    WrapperAssert(dest.is_same_size(src), PyExc_ValueError,
                  "`dest` and `src` must be the same size.");
    
    WrapperAssert(!dest.is_cuda(), PyExc_ValueError,
                  "`dest` must be a CPU tensor.");
    WrapperAssert(src.is_cuda(), PyExc_ValueError,
                  "`src` must be a CUDA tensor.");
    
    WrapperAssert(dest.is_contiguous(), PyExc_ValueError,
                  "`dest` must be a contiguous tensor.");
    WrapperAssert(src.is_contiguous(), PyExc_ValueError,
                  "`src` must be a contiguous tensor.");
    
    WrapperAssert(dest.type().scalarType() == at::ScalarType::Float, PyExc_ValueError,
                  "`dest` must be a float32 tensor.");
    WrapperAssert(src.type().scalarType() == at::ScalarType::Float, PyExc_ValueError,
                  "`src` must be a float32 tensor.");

    // perform the memcpy
    const size_t numel = src.numel();
    void* dest_data = dest.data_ptr();
    void* src_data = src.data_ptr();

    cudaMemcpy(dest_data, src_data, numel*sizeof(float), cudaMemcpyDeviceToHost);
    
    Py_RETURN_NONE;
}
