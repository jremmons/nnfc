#include <Python.h>
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL nnfc_codec_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <iostream>

#include "common.hh"
#include "nnfc_decoder.hh"

static std::vector<std::vector<uint8_t>> pylist2buffers(PyObject* input_pylist) {

    Py_ssize_t length = PyList_Size(input_pylist);
    std::vector<std::vector<uint8_t>> buffers;

    for(Py_ssize_t i = 0; i < length; i++) {
        PyArrayObject *array = reinterpret_cast<PyArrayObject*>(PyList_GetItem(input_pylist, i));

        WrapperAssert(PyArray_Check(array), PyExc_TypeError, "the elements of the input python list must be uint8 numpy arrays.");
        WrapperAssert(PyArray_TYPE(array) == NPY_UINT8, PyExc_TypeError, "the elements of the input python list must be uint8 numpy arrays.");

        size_t nElements = PyArray_SIZE(array);
        //size_t stride0 = PyArray_STRIDE(array, 0);
        //WrapperAssert(nElements == stride0, PyExc_RuntimeError, "nElements==stride0: " + std::to_string(nElements) + std::string("==") + std::to_string(stride0));
        uint8_t *data = static_cast<uint8_t*>(PyArray_DATA(array));

        std::vector<uint8_t> buffer(nElements);
        std::memcpy(buffer.data(), data, nElements);

        buffers.push_back(buffer);
    }
    
    return buffers;
}

static PyObject* tensors2blob(std::vector<NN::Tensor<float, 3>> input_tensors) {

    npy_intp num_tensors = input_tensors.size();
    if(num_tensors == 0){
        PyErr_SetString(PyExc_RuntimeError, "No input tensors! (this is a library bug)");
        PyErr_Print();
    }

    npy_intp dim0 = input_tensors[0].dimension(0);
    npy_intp dim1 = input_tensors[0].dimension(1);
    npy_intp dim2 = input_tensors[0].dimension(2);
    const npy_intp tensor_size = dim0*dim1*dim2;
    
    for(npy_intp i = 0; i < num_tensors; i++) {
        WrapperAssert(input_tensors[i].dimension(0) == dim0, PyExc_TypeError, "decoded tensors dimensions do not match.");
        WrapperAssert(input_tensors[i].dimension(1) == dim1, PyExc_TypeError, "decoded tensors dimensions do not match.");
        WrapperAssert(input_tensors[i].dimension(2) == dim2, PyExc_TypeError, "decoded tensors dimensions do not match.");
        WrapperAssert(input_tensors[i].size() == tensor_size, PyExc_TypeError, "decoded tensors size is not correct.");
    }

    npy_intp array_size[4] = { num_tensors, dim0, dim1, dim2 };
    PyArrayObject *array = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(4, array_size, NPY_FLOAT32));
    WrapperAssert(PyArray_ISCARRAY(array), PyExc_TypeError, "the output array must be a c-style array and conriguous in memory. (this is a library bug)");

    float *array_data = static_cast<float*>(PyArray_DATA(array));
        
    // TODO(jremmons) do a smarter memcpy. The current implementation
    // assumes the data is contiguous in memory and in c-style, but we
    // should check this before doing a memcpy! 
    for(npy_intp i = 0; i < num_tensors; i++) {
        auto tensor = input_tensors[i];
        std::memcpy(&array_data[tensor_size * i], &tensor(0,0,0), sizeof(float)*tensor_size);
    }
    
    return reinterpret_cast<PyObject*>(array);
}


PyObject* NNFCDecoderContext_new(PyTypeObject *type, PyObject *, PyObject *) {

    NNFCDecoderContext *self;

    self = (NNFCDecoderContext*)type->tp_alloc(type, 0);
    if(self == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Could not alloc a new NNFCDecoderContext.");
        PyErr_Print();
    }

    return (PyObject*) self;
}

void NNFCDecoderContext_dealloc(NNFCDecoderContext* self) {

    delete self->decoder;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

int NNFCDecoderContext_init(NNFCDecoderContext *self, PyObject *args, PyObject *) {

    char *codec_name = NULL;
    if (!PyArg_ParseTuple(args, "s", &codec_name)){
        return 0;
    }

    self->decoder = new NNFC::SimpleDecoder();
    return 0;

}

PyObject* NNFCDecoderContext_decode(NNFCDecoderContext *self, PyObject *args){

    PyObject *input_pylist;

    if (!PyArg_ParseTuple(args, "O", &input_pylist)){
        PyErr_Print();
    }

    if(!PyList_Check(input_pylist)) {
        PyErr_SetString(PyExc_TypeError, "the input should be a python list of numpy array");
        PyErr_Print();
    }
    
    try {
        std::vector<std::vector<uint8_t>> input_buffers = pylist2buffers(input_pylist);

        std::vector<NN::Tensor<float, 3>> tensors;
        
        for(size_t i = 0; i < input_buffers.size(); i++) {

            NN::Tensor<float, 3> tensor = self->decoder->decode(input_buffers[i]);

            tensors.push_back(tensor);
        }

        PyObject *array = tensors2blob(tensors);
        return array;
        
    }
    catch(nnfc_python_exception e) {
        PyErr_SetString(e.type(), e.what());
        return 0;
    }
    catch(std::exception e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return 0;
    }

}

PyObject* NNFCDecoderContext_backprop(NNFCDecoderContext *, PyObject *){

    Py_RETURN_NONE;
}

