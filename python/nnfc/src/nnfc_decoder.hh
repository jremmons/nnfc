#ifndef _NNFC_DECODER
#define _NNFC_DECODER
#include <Python.h>

#include "nnfc/nnfc_CXXAPI.hh"

typedef struct {

    PyObject_HEAD
    std::unique_ptr<nnfc::cxxapi::DecoderContextInterface> decoder;
    
} NNFCDecoderContext;

PyObject* NNFCDecoderContext_new(PyTypeObject *type, PyObject *, PyObject *);
int NNFCDecoderContext_init(NNFCDecoderContext *self, PyObject *args, PyObject *);
void NNFCDecoderContext_dealloc(NNFCDecoderContext* self);

PyObject* NNFCDecoderContext_forward(NNFCDecoderContext *self, PyObject *args);
PyObject* NNFCDecoderContext_backward(NNFCDecoderContext *self, PyObject *args);

#endif // _NNFC_DECODER
