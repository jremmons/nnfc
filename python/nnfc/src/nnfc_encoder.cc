extern "C" {
#include <Python.h>
#include <TH/TH.h>
}

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

int NNFCEncoderContext_init(NNFCEncoderContext *self, PyObject *args, PyObject *kwargs) {

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

    THFloatTensor *input;

    if (!PyArg_ParseTuple(args, "O",
                          &input)){
        return 0;
    }

    // try {
    //     self->counter->start();
    // }
    // catch(const std::exception& e) {
    //     PyErr_SetString(PyExc_ValueError, e.what());
    //     return 0;
    // }        

    PyObject *val = PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(42));        
    return val;
}

PyObject* NNFCEncoderContext_backprop(NNFCEncoderContext *self, PyObject *args){

    THFloatTensor *input;

    if (!PyArg_ParseTuple(args, "O",
                          &input)){
        return 0;
    }

    // try {
    //     self->counter->start();
    // }
    // catch(const std::exception& e) {
    //     PyErr_SetString(PyExc_ValueError, e.what());
    //     return 0;
    // }        

    PyObject *val = PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(42));        
    return val;
}

