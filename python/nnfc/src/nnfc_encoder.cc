extern "C" {
#include <Python.h>
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL nnfc_codec_ARRAY_API
#include <numpy/arrayobject.h>
}

#include <iostream>
#include <string>

//#include ""
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

    // char *counter_name = NULL;

    // static char *kwlist[] = { "counter_name", NULL };
    // if (!PyArg_ParseTupleAndKeywords(args,
    //                                  kwargs,
    //                                  "s",
    //                                  kwlist,
    //                                  &counter_name)){

    //     PyErr_SetString(PyExc_ValueError, "NNFCEncoderContext failed while parsing constructor args/kwargs.");
    //     return 0;
    // }

    // if(counter_name == NULL) {

    //     PyErr_SetString(PyExc_ValueError, "NNFCEncoderContext requires `counter_name` to be specified.");
    //     return 0;
    // }

    // // std::cerr << "got it: '" << counter_name << "'\n";
    // try {

    //     libperf::NNFCEncoderContext *p = new libperf::NNFCEncoderContext{std::string(counter_name)};
    //     self->counter = p;
    // }
    // catch(const std::exception& e) {

    //     std::string error_message = std::string(e.what());
    //     error_message += std::string(" Try running `nnfc_codec.get_available_counters()` to list the available counters on your system.");
    //     PyErr_SetString(PyExc_ValueError, error_message.c_str());
    //     return 0;
    // }

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
    input_array = PyArray_GETCONTIGUOUS(input_array);
    if(!input_array) {
        PyErr_SetString(PyExc_ValueError,  "could not convert input to an array with a contiguous memory layout.");
        return 0;        
    }
    
    const int ndims = PyArray_NDIM(input_array);
    if(ndims != 4) {
        std::string error_message("the input to the encoder must be a 4D numpy.ndarray. (The input dimensionality was: ");
        error_message = error_message + std::to_string(ndims) + std::string(")");
        PyErr_SetString(PyExc_ValueError,  error_message.c_str());
        return 0;
    }

    
    
    // TODO(jremmons) call down to the libnnfc encode function
    // TODO(jremmons) munge the output data so that 

    
    PyArrayObject *output_array = input_array;
    return reinterpret_cast<PyObject*>(output_array);
}

PyObject* NNFCEncoderContext_backprop(NNFCEncoderContext *self, PyObject *args){

    Py_RETURN_NONE;
}

