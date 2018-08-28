#ifndef _NNFC_ENCODER
#define _NNFC_ENCODER
#include <Python.h>

#include "nnfc/nnfc_CXXAPI.hh"

typedef struct {

    PyObject_HEAD
    std::unique_ptr<nnfc::cxxapi::EncoderContextInterface> encoder;
    
} NNFCEncoderContext;

PyObject* NNFCEncoderContext_new(PyTypeObject *type, PyObject *, PyObject *);
int NNFCEncoderContext_init(NNFCEncoderContext *self, PyObject *args, PyObject *);
void NNFCEncoderContext_dealloc(NNFCEncoderContext* self);

PyObject* NNFCEncoderContext_forward(NNFCEncoderContext *self, PyObject *args);
PyObject* NNFCEncoderContext_backward(NNFCEncoderContext *self, PyObject *args);

#endif // _NNFC_ENCODER
