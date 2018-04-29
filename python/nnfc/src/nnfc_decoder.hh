#ifndef _NNFC_DECODER
#define _NNFC_DECODER

extern "C" {
#include <Python.h>
}

#include "nnfc.hh"

typedef struct {

    PyObject_HEAD
    nnfc::SimpleDecoder *decoder;
    
} NNFCDecoderContext;

PyObject* NNFCDecoderContext_new(PyTypeObject *type, PyObject *, PyObject *);
int NNFCDecoderContext_init(NNFCDecoderContext *self, PyObject *args, PyObject *);
void NNFCDecoderContext_dealloc(NNFCDecoderContext* self);

PyObject* NNFCDecoderContext_decode(NNFCDecoderContext *self, PyObject *args);
PyObject* NNFCDecoderContext_backprop(NNFCDecoderContext *self, PyObject *args);

#endif // _NNFC_DECODER
