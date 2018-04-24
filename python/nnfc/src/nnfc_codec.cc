#include <Python.h>
#include <TH/TH.h>

#include "nnfc_encoder.hh"
#include "nnfc_decoder.hh"

#ifdef _NNFC_CUDA_AVAILABLE
#include <THC/THC.h>
#endif

extern "C" {

    // define the NNFCEncoderContext /////////////////////////////////
    static PyMethodDef NNFCEncoderContext_methods[] = {
        {"encode", (PyCFunction)NNFCEncoderContext_encode, METH_VARARGS,
         "Encodes a batch of intermediate activations. Expects a PyTorch float tensor as input."
        },
        {"backprop", (PyCFunction)NNFCEncoderContext_backprop, METH_VARARGS,
         "Propagates gradients back through this operation."
        },
        {NULL, NULL, METH_VARARGS, ""}  // Sentinel
    };
    
    static PyTypeObject NNFCEncoderContextType = {
        PyObject_HEAD_INIT(NULL)
        .tp_name = "nnfc_codec.EncoderContext",
        .tp_basicsize = sizeof(NNFCEncoderContext),
        .tp_itemsize = 0,
        .tp_dealloc = (destructor)NNFCEncoderContext_dealloc,
        .tp_print = 0,
        .tp_getattr = 0,
        .tp_setattr = 0,
        .tp_as_async = 0,
        .tp_repr = 0,
        .tp_as_number = 0,
        .tp_as_sequence = 0,
        .tp_as_mapping = 0,
        .tp_hash = 0,
        .tp_call = 0,
        .tp_str = 0,
        .tp_getattro = 0,
        .tp_setattro = 0,
        .tp_as_buffer = 0,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "EncoderContext object",
        .tp_traverse = 0,
        .tp_clear = 0,
        .tp_richcompare = 0,
        .tp_weaklistoffset = 0,
        .tp_iter = 0,
        .tp_iternext = 0,
        .tp_methods = NNFCEncoderContext_methods,
        .tp_members = 0,
        .tp_getset = 0,
        .tp_base = 0,
        .tp_dict = 0,
        .tp_descr_get = 0,
        .tp_descr_set = 0,
        .tp_dictoffset = 0,
        .tp_init = (initproc)NNFCEncoderContext_init,
        .tp_alloc = 0,
        .tp_new = NNFCEncoderContext_new,
        .tp_free = 0,
        .tp_is_gc = 0,
        .tp_bases = 0,
        .tp_mro = 0,
        .tp_cache = 0,
        .tp_subclasses = 0,
        .tp_weaklist = 0,
        .tp_del = 0,
        .tp_version_tag = 0,
        .tp_finalize = 0,
    };

    // define the NNFCDecoderContext /////////////////////////////////
    static PyMethodDef NNFCDecoderContext_methods[] = {
        {"decode", (PyCFunction)NNFCDecoderContext_decode, METH_VARARGS,
         "Decodes a batch of intermediate activations. Expects a PyTorch byte tensor as input."
        },
        {"backprop", (PyCFunction)NNFCDecoderContext_backprop, METH_VARARGS,
         "Propagates gradients back through this operation."
        },
        {NULL, NULL, METH_VARARGS, ""}  // Sentinel
    };
    
    static PyTypeObject NNFCDecoderContextType = {
        PyObject_HEAD_INIT(NULL)
        .tp_name = "nnfc_codec.NNFCDecoderContext",
        .tp_basicsize = sizeof(NNFCDecoderContext),
        .tp_itemsize = 0,
        .tp_dealloc = (destructor)NNFCDecoderContext_dealloc,
        .tp_print = 0,
        .tp_getattr = 0,
        .tp_setattr = 0,
        .tp_as_async = 0,
        .tp_repr = 0,
        .tp_as_number = 0,
        .tp_as_sequence = 0,
        .tp_as_mapping = 0,
        .tp_hash = 0,
        .tp_call = 0,
        .tp_str = 0,
        .tp_getattro = 0,
        .tp_setattro = 0,
        .tp_as_buffer = 0,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = "NNFCDecoderContext object",
        .tp_traverse = 0,
        .tp_clear = 0,
        .tp_richcompare = 0,
        .tp_weaklistoffset = 0,
        .tp_iter = 0,
        .tp_iternext = 0,
        .tp_methods = NNFCDecoderContext_methods,
        .tp_members = 0,
        .tp_getset = 0,
        .tp_base = 0,
        .tp_dict = 0,
        .tp_descr_get = 0,
        .tp_descr_set = 0,
        .tp_dictoffset = 0,
        .tp_init = (initproc)NNFCDecoderContext_init,
        .tp_alloc = 0,
        .tp_new = NNFCDecoderContext_new,
        .tp_free = 0,
        .tp_is_gc = 0,
        .tp_bases = 0,
        .tp_mro = 0,
        .tp_cache = 0,
        .tp_subclasses = 0,
        .tp_weaklist = 0,
        .tp_del = 0,
        .tp_version_tag = 0,
        .tp_finalize = 0,
    };

    // define the nnfc_codec extension module  ///////////////////////
    
    struct module_state {
        PyObject *error;
    };

    // static PyObject* nnfc_codec_encode(PyObject *, PyObject *args) {
    //     PyObject *hello = PyUnicode_FromString("hello world");
    //     return hello;
    // }

    // static PyObject* nnfc_codec_decode(PyObject *args) {
    //     PyObject *hello = PyUnicode_FromString("hello world");
    //     return hello;
    // }

    
    static PyMethodDef module_methods[] = {
        // { "encode", (PyCFunction)nnfc_codec_encode, METH_VARARGS, "encode a batch of intermediate activations" },
        // { "decode", (PyCFunction)nnfc_codec_decode, METH_VARARGS, "decode a batch of intermediate activations" },
        {NULL, NULL, 0, NULL}  /* Sentinel */
    };
    
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        .m_name = "nnfc_codec",
        .m_doc = "a python library for accessing CPU performance counters on linux.",
        .m_size = sizeof(struct module_state),
        .m_methods = module_methods,
        .m_slots = 0,
        .m_traverse = 0,
        .m_clear = 0,
        .m_free = 0
    };
    
    PyMODINIT_FUNC
    PyInit_nnfc_codec(void)
    {

        // set global module functions and parameters
        PyObject* module = PyModule_Create(&module_def);

        if (module == NULL){
            PyErr_SetString(PyExc_ValueError, "could not create nnfc_codec module.");
            return 0;
        }

        PyObject *module_namespace = PyModule_GetDict(module);
        PyObject *version = PyUnicode_FromString(_NNFC_VERSION);

        if(PyDict_SetItemString(module_namespace, "__version__", version)){
            PyErr_SetString(PyExc_ValueError, "could not set __version__ of nnfc_codec module.");
            return 0;
        }

        // add the NNFCEncoderContext python class to the nnfc_codec module
        if (PyType_Ready(&NNFCEncoderContextType) < 0){
            PyErr_SetString(PyExc_ValueError, "could not intialize NNFCEncoderContext object.");
            return 0;
        }

        Py_INCREF(&NNFCEncoderContextType);
        PyModule_AddObject(module, "EncoderContext", (PyObject *)&NNFCEncoderContextType);

        // add the NNFCDecoderContext python class to the nnfc_codec module
        if (PyType_Ready(&NNFCDecoderContextType) < 0){
            PyErr_SetString(PyExc_ValueError, "could not intialize NNFCDecoderContext object.");
            return 0;
        }

        Py_INCREF(&NNFCDecoderContextType);
        PyModule_AddObject(module, "DecoderContext", (PyObject *)&NNFCDecoderContextType);

        return module;
    }

}
