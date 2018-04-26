#ifndef _NNFC_CUDA
#define _NNFC_CUDA

extern "C" {
#include <Python.h>
}

PyObject* NNFCinplace_copy_d2h(PyObject *self, PyObject *args, PyObject *kwargs);

#endif // _NNFC_CUDA
