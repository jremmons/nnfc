#ifndef _NNFC_ENCODER
#define _NNFC_ENCODER

#include <Python.h>
    
typedef struct {

    PyObject_HEAD
    
} NNFCEncoderContext;

PyObject* NNFCEncoderContext_new(PyTypeObject *type, PyObject *, PyObject *);
int NNFCEncoderContext_init(NNFCEncoderContext *self, PyObject *args, PyObject *);
void NNFCEncoderContext_dealloc(NNFCEncoderContext* self);

PyObject* NNFCEncoderContext_encode(NNFCEncoderContext *self, PyObject *args);
PyObject* NNFCEncoderContext_backprop(NNFCEncoderContext *self, PyObject *args);

#endif // _NNFC_ENCODER
