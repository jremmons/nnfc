extern "C" {
#include <Python.h>
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL nnfc_codec_ARRAY_API
#include <numpy/arrayobject.h>
}

#include <iostream>
#include <string>

// TOOD(jremmons) use the stable API exposed by nnfc (once ready)
// rather than trying to compile against the internal header files
// that might change quickly. 
//#include "nnfc_API.hh"
#include "nnfc.hh"

#include "nnfc_encoder.hh"

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

    Py_TYPE(self)->tp_free((PyObject*)self);
}

int NNFCEncoderContext_init(NNFCEncoderContext *self, PyObject *args, PyObject *) {

    char *codec_name = NULL;
    if (!PyArg_ParseTuple(args, "s", &codec_name)){
        return 0;
    }
        
    return 0;
}

PyObject* NNFCEncoderContext_encode(NNFCEncoderContext *self, PyObject *args){

    PyArrayObject *input_array;

    {
        PyObject *input;
        if (!PyArg_ParseTuple(args, "O", &input)){
            return 0;
        }
        
        if(!PyArray_Check(input)) {
            PyErr_SetString(PyExc_ValueError, "the input to the encoder must be a 4D numpy.ndarray.");
            return 0;
        }

        input_array = reinterpret_cast<PyArrayObject*>(input);
    }

    // TODO(jremmons) we probably don't need to force the arrays to be contiguous 
    // input_array = PyArray_GETCONTIGUOUS(input_array);
    // if(!input_array) {
    //     PyErr_SetString(PyExc_ValueError,  "could not convert input to an array with a contiguous memory layout.");
    //     return 0;        
    // }
    
    // const int ndims = PyArray_NDIM(input_array);
    // if(ndims != 4) {
    //     std::string error_message("the input to the encoder must be a 4D numpy.ndarray. (The input dimensionality was: ");
    //     error_message = error_message + std::to_string(ndims) + std::string(")");
    //     PyErr_SetString(PyExc_ValueError,  error_message.c_str());
    //     return 0;
    // }
    
    // TODO(jremmons) call down to the libnnfc encode function
    // TODO(jremmons) munge the output data so that 


    Py_INCREF(input_array);
    PyArrayObject *output_array = input_array;
    return reinterpret_cast<PyObject*>(output_array);
}

PyObject* NNFCEncoderContext_backprop(NNFCEncoderContext *self, PyObject *args){

    Py_RETURN_NONE;
}

