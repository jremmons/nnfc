extern "C" {
#include <Python.h>
#include <TH/TH.h>
}

#include "nnfc_decoder.hh"
   
extern "C" PyObject* NNFCDecoderContext_new(PyTypeObject *type, PyObject *, PyObject *) {

    NNFCDecoderContext *self;

    self = (NNFCDecoderContext*)type->tp_alloc(type, 0);
    if(self == NULL) {
        PyErr_SetString(PyExc_ValueError, "Could not alloc a new NNFCDecoderContext.");
        return 0;
    }

    return (PyObject*) self;
}

extern "C" void NNFCDecoderContext_dealloc(NNFCDecoderContext* self) {

    Py_TYPE(self)->tp_free((PyObject*)self);
}

extern "C" int NNFCDecoderContext_init(NNFCDecoderContext *self, PyObject *args, PyObject *kwargs) {

    // char *counter_name = NULL;

    // static char *kwlist[] = { "counter_name", NULL };
    // if (!PyArg_ParseTupleAndKeywords(args,
    //                                  kwargs,
    //                                  "s",
    //                                  kwlist,
    //                                  &counter_name)){

    //     PyErr_SetString(PyExc_ValueError, "NNFCDecoderContext failed while parsing constructor args/kwargs.");
    //     return 0;
    // }

    // if(counter_name == NULL) {

    //     PyErr_SetString(PyExc_ValueError, "NNFCDecoderContext requires `counter_name` to be specified.");
    //     return 0;
    // }

    // // std::cerr << "got it: '" << counter_name << "'\n";
    // try {

    //     libperf::NNFCDecoderContext *p = new libperf::NNFCDecoderContext{std::string(counter_name)};
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

extern "C" PyObject* NNFCDecoderContext_decode(NNFCDecoderContext *self, PyObject *args){

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

extern "C" PyObject* NNFCDecoderContext_backprop(NNFCDecoderContext *self, PyObject *args){

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

