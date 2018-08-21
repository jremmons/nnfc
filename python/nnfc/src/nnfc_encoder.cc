#include <Python.h>
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL nnfc_codec_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <omp.h>

#include <exception>
#include <iostream>
#include <string>

#include "nnfc_CXXAPI.hh"
#include "common.hh"
#include "tensor.hh"

#include "nnfc_encoder.hh"

static std::vector<nn::Tensor<float, 3>> blob2tensors(PyArrayObject *input_array) {

    WrapperAssert(PyArray_ISCARRAY(input_array), PyExc_TypeError, "the input array must be a c-style array and contiguous in memory.");
    WrapperAssert(PyArray_TYPE(input_array) == NPY_FLOAT32, PyExc_TypeError, "the input array must have dtype==float32.");

    const int ndims = PyArray_NDIM(input_array);
    WrapperAssert(ndims == 4, PyExc_TypeError, std::string("the input to the encoder must be a 4D numpy.ndarray. (The input dimensionality was: ") + std::to_string(ndims) + std::string(")"));

    Eigen::Index dim0 = PyArray_DIM(input_array, 0);
    Eigen::Index dim1 = PyArray_DIM(input_array, 1);
    Eigen::Index dim2 = PyArray_DIM(input_array, 2);
    Eigen::Index dim3 = PyArray_DIM(input_array, 3);

    size_t stride0 = PyArray_STRIDE(input_array, 0);
    size_t nStride0 = stride0 / sizeof(float);
    size_t nElements = PyArray_SIZE(input_array);
        
    float *data = static_cast<float*>(PyArray_DATA(input_array));

    std::vector<nn::Tensor<float, 3>> tensors;
    tensors.reserve(dim0);
    
    for(size_t offset = 0; offset < nElements; offset += nStride0) {
        tensors.emplace_back(&data[offset], dim1, dim2, dim3);
    }

    return tensors;
}

static PyObject* buffers2pylist(std::vector<std::vector<uint8_t>> input_arrays) {

    Py_ssize_t num_arrays = input_arrays.size();

    PyObject *pylist = PyList_New(num_arrays);

    for(Py_ssize_t i = 0; i < num_arrays; i++) {
        
        npy_intp size = input_arrays[i].size();
        PyObject *array = PyArray_SimpleNew(1, &size, NPY_UBYTE);

        uint8_t *data = static_cast<uint8_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)));
        std::memcpy(data, input_arrays[i].data(), size);
        
        if(PyList_SetItem(pylist, i, array)) {
            PyErr_SetString(PyExc_RuntimeError, "could not insert buffer into python list.");
            PyErr_Print();            
        }
    }
    
    return pylist;
}

PyObject* NNFCEncoderContext_new(PyTypeObject *type, PyObject *, PyObject *) {

    NNFCEncoderContext *self;

    self = (NNFCEncoderContext*)type->tp_alloc(type, 0);
    if(self == NULL) {
        PyErr_SetString(PyExc_ValueError, "Could not alloc a new NNFCEncoderContext.");
        return 0;
    }

    return (PyObject*) self;
}

void NNFCEncoderContext_dealloc(NNFCEncoderContext* self) {

    self->encoder.release();
    Py_TYPE(self)->tp_free((PyObject*)self);
}

int NNFCEncoderContext_init(NNFCEncoderContext *self, PyObject *args, PyObject*) {

    char *encoder_name = nullptr;
    PyObject* dictionary_args = nullptr;
    if (!PyArg_ParseTuple(args, "sO", &encoder_name, &dictionary_args)) {
        return 0;
    }

    try {
        const auto encoder_arg_types = nnfc::cxxapi::get_encoder_constructor_types(encoder_name);
        
        WrapperAssert(PyDict_Check(dictionary_args),
                      PyExc_TypeError,
                      "second argument must be a dictionary with the corrsponding constructor parameters.");

        const auto encoder_args = parse_dict(dictionary_args, encoder_arg_types);
        
        self->encoder = std::move(nnfc::cxxapi::new_encoder(encoder_name, encoder_args));
    }
    catch(nnfc_python_exception& e) {
        PyErr_SetString(e.type(), e.what());
        return 0;
    }
    catch(std::exception& e) {
        std::string error_message = e.what();
        PyErr_SetString(PyExc_Exception, error_message.c_str());
        return 0;
    }
    
    return 0;
}

PyObject* NNFCEncoderContext_forward(NNFCEncoderContext *self, PyObject *args){

    PyArrayObject *input_array;

    if (!PyArg_ParseTuple(args, "O", &input_array)){
        PyErr_Print();
    }

    if(!PyArray_Check(input_array)) {
        PyErr_SetString(PyExc_TypeError, "the input to the encoder must be a 4D numpy.ndarray.");
        PyErr_Print();
    }

    try {
        std::vector<nn::Tensor<float, 3>> input_tensors = blob2tensors(input_array);
        const size_t input_tensors_size = input_tensors.size();
        std::vector<std::vector<uint8_t>> buffers(input_tensors_size);

        // #pragma omp parallel for
        // for(size_t i = 0; i < input_tensors_size; i++) {
        //     const std::vector<uint8_t> buffer = self->encoder->forward(input_tensors[i]);
        //     buffers[i] = buffer;
        // }

        for(size_t i = 0; i < input_tensors_size; i++) {
            const std::vector<uint8_t> buffer = self->encoder->forward(input_tensors[i]);
            buffers[i] = buffer;
        }

        PyObject *pylist_of_buffer = buffers2pylist(buffers);
        return pylist_of_buffer;

    }
    catch(nnfc_python_exception& e) {
        PyErr_SetString(e.type(), e.what());
        return 0;
    }
    catch(std::exception& e) {
        std::string error_message = e.what();
        PyErr_SetString(PyExc_Exception, error_message.c_str());
        return 0;
   }
    
}

PyObject* NNFCEncoderContext_backward(NNFCEncoderContext *, PyObject *args){

    PyObject* grad_output = nullptr;
        
    if (!PyArg_ParseTuple(args, "O", &grad_output)){
        PyErr_Print();
    }

    return grad_output;
}

