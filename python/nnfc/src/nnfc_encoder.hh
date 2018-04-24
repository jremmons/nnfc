#ifndef _NNFC_ENCODER
#define _NNFC_ENCODER

extern "C" {
#include <Python.h>
}
    
typedef struct {

    PyObject_HEAD
    
} NNFCEncoderContext;

extern "C" PyObject* NNFCEncoderContext_new(PyTypeObject *type, PyObject *, PyObject *);
extern "C" int NNFCEncoderContext_init(NNFCEncoderContext *self, PyObject *args, PyObject *);
extern "C" void NNFCEncoderContext_dealloc(NNFCEncoderContext* self);

extern "C" PyObject* NNFCEncoderContext_encode(NNFCEncoderContext *self, PyObject *args);
extern "C" PyObject* NNFCEncoderContext_backprop(NNFCEncoderContext *self, PyObject *args);

#endif // _NNFC_ENCODER
