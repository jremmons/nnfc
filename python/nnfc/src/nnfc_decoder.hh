#ifndef _NNFC_DECODER
#define _NNFC_DECODER

#include <Python.h>

typedef struct {

    PyObject_HEAD
    
} NNFCDecoderContext;

extern "C" PyObject* NNFCDecoderContext_new(PyTypeObject *type, PyObject *, PyObject *);
extern "C" int NNFCDecoderContext_init(NNFCDecoderContext *self, PyObject *args, PyObject *);
extern "C" void NNFCDecoderContext_dealloc(NNFCDecoderContext* self);

extern "C" PyObject* NNFCDecoderContext_decode(NNFCDecoderContext *self, PyObject *args);
extern "C" PyObject* NNFCDecoderContext_backprop(NNFCDecoderContext *self, PyObject *args);

#endif // _NNFC_DECODER
