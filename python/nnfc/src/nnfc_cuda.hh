#ifndef _NNFC_CUDA
#define _NNFC_CUDA

extern "C" {
#include <Python.h>
}

PyObject* NNFCtensor_memcpy_d2h(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject* NNFCtensor_memcpy_h2d(PyObject *self, PyObject *args, PyObject *kwargs);

#endif // _NNFC_CUDA
