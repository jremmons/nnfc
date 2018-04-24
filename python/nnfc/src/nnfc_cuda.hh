#ifndef _NNFC_CUDA
#define _NNFC_CUDA

extern "C" {
#include <Python.h>
}

extern "C" PyObject* device_to_host_copy(PyObject*, PyObject *args, PyObject *);

#endif // _NNFC_CUDA
