extern "C" {
#include <Python.h>
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL nnfc_codec_ARRAY_API
#include <numpy/arrayobject.h>
}
    
#include "nnfc_decoder.hh"
   
PyObject* NNFCDecoderContext_new(PyTypeObject *type, PyObject *, PyObject *) {

    NNFCDecoderContext *self;

    self = (NNFCDecoderContext*)type->tp_alloc(type, 0);
    if(self == NULL) {
        PyErr_SetString(PyExc_ValueError, "Could not alloc a new NNFCDecoderContext.");
        return 0;
    }

    return (PyObject*) self;
}

void NNFCDecoderContext_dealloc(NNFCDecoderContext* self) {

    //Py_TYPE(self)->tp_free((PyObject*)self);
}

int NNFCDecoderContext_init(NNFCDecoderContext *self, PyObject *args, PyObject *kwargs) {

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

PyObject* NNFCDecoderContext_decode(NNFCDecoderContext *self, PyObject *args){

    //torch::THPVoidTensor *input;
    PyObject * input;
    
    if (!PyArg_ParseTuple(args, "O",
                          &input)){
        return 0;
    }

    // torch::THPVoidTensor THPVoidTensorType;
    // torch::THPVoidTensor *output = PyObject_New(torch::THPVoidTensor, (PyTypeObject*)&THPVoidTensorType);

    // THFloatTensor *input_tensor = reinterpret_cast<THFloatTensor*>(input->cdata);
    // THFloatTensor *output_tensor = reinterpret_cast<THFloatTensor*>(output->cdata);
    
    // THFloatTensor_resizeAs(input_tensor, output_tensor);

    // int nElements =  THFloatTensor_nElement(input_tensor);

    // float *input_data = THFloatTensor_data(input_tensor);
    // float *output_data = THFloatTensor_data(output_tensor);
    // for(int i = 0; i < nElements; i++){
    //     output_data[i] = input_data[i];
    // }

    Py_INCREF(input);
    
    return (PyObject *) input;
}

PyObject* NNFCDecoderContext_backprop(NNFCDecoderContext *self, PyObject *args){

    PyObject * input;

    if (!PyArg_ParseTuple(args, "O",
                          &input)){
        return 0;
    }
    
    //torch::THPVoidTensor *output; //= PyObject_New(torch::THPVoidTensor, &torch::THPVoidTensor);
    //output = PyObject_Init(output, &torch::THPVoidTensor);

    // THFloatTensor *output = THFloatTensor_new();
    // THFloatTensor_resizeAs(input, output);

    // int nElements =  THFloatTensor_nElement(input);

    // float *input_data = THFloatTensor_data(input);
    // float *output_data = THFloatTensor_data(input);
    // for(int i = 0; i < nElements; i++){
    //     output_data[i] = input_data[i];
    // }

    Py_INCREF(input);
    
    return (PyObject *) input;
}

