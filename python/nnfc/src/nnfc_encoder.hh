#ifndef _NNFC_ENCODER
#define _NNFC_ENCODER
extern "C" {
#include <Python.h>
}

#include "nnfc.hh"

typedef struct {

    PyObject_HEAD
    nnfc::SimpleEncoder *encoder;
    
} NNFCEncoderContext;

PyObject* NNFCEncoderContext_new(PyTypeObject *type, PyObject *, PyObject *);
int NNFCEncoderContext_init(NNFCEncoderContext *self, PyObject *args, PyObject *);
void NNFCEncoderContext_dealloc(NNFCEncoderContext* self);

PyObject* NNFCEncoderContext_encode(NNFCEncoderContext *self, PyObject *args);
PyObject* NNFCEncoderContext_backprop(NNFCEncoderContext *self, PyObject *args);

#endif // _NNFC_ENCODER
