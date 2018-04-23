#include <Python.h>

#include <TH/TH.h>
#include <THC/THC.h>

extern "C" {

    typedef struct {

        PyObject_HEAD

    } NNFCCodecState;

    static PyObject*
    NNFCCodecState_new(PyTypeObject *type, PyObject *, PyObject *) {

        NNFCCodecState *self;

        self = (NNFCCodecState*)type->tp_alloc(type, 0);
        if(self == NULL) {
            PyErr_SetString(PyExc_ValueError, "Could not alloc a new NNFCCodecState.");
            return 0;
        }

        return (PyObject*) self;
    }

    static void
    NNFCCodecState_dealloc(NNFCCodecState* self) {

        Py_TYPE(self)->tp_free((PyObject*)self);
    }
    
    static int
    NNFCCodecState_init(NNFCCodecState *self, PyObject *args, PyObject *kwargs) {

        // char *counter_name = NULL;

        // static char *kwlist[] = { "counter_name", NULL };
        // if (!PyArg_ParseTupleAndKeywords(args,
        //                                  kwargs,
        //                                  "s",
        //                                  kwlist,
        //                                  &counter_name)){

        //     PyErr_SetString(PyExc_ValueError, "NNFCCodecState failed while parsing constructor args/kwargs.");
        //     return 0;
        // }

        // if(counter_name == NULL) {

        //     PyErr_SetString(PyExc_ValueError, "NNFCCodecState requires `counter_name` to be specified.");
        //     return 0;
        // }

        // // std::cerr << "got it: '" << counter_name << "'\n";
        // try {

        //     libperf::NNFCCodecState *p = new libperf::NNFCCodecState{std::string(counter_name)};
        //     self->counter = p;
        // }
        // catch(const std::exception& e) {

        //     std::string error_message = std::string(e.what());
        //     error_message += std::string(" Try running `nnfc_codec.get_available_counters()` to list the available counters on your system.");
        //     PyErr_SetString(PyExc_ValueError, error_message.c_str());
        //     return 0;
        // }
        
        return 0;
    }

    static PyObject*
    NNFCCodecState_start(NNFCCodecState *self){

        // try {
        //     self->counter->start();
        // }
        // catch(const std::exception& e) {
        //     PyErr_SetString(PyExc_ValueError, e.what());
        //     return 0;
        // }        

        Py_RETURN_NONE;
    }

    static PyObject*
    NNFCCodecState_stop(NNFCCodecState *self){

        // try {
        //     self->counter->stop();
        // }
        // catch(const std::exception& e) {
        //     PyErr_SetString(PyExc_ValueError, e.what());
        //     return 0;
        // }

        Py_RETURN_NONE;
    }

    static PyObject*
    NNFCCodecState_reset(NNFCCodecState *self){

        // try {
        //     self->counter->reset();       
        // }
        // catch(const std::exception& e) {
        //     PyErr_SetString(PyExc_ValueError, e.what());
        //     return 0;
        // }

        Py_RETURN_NONE;
    }

    static PyObject*
    NNFCCodecState_getval(NNFCCodecState *self){

        // uint64_t counter_val;
        // try {
        //     counter_val = self->counter->getval();
        // }
        // catch(const std::exception& e) {
        //     PyErr_SetString(PyExc_ValueError, e.what());
        //     return 0;
        // }

        // static_assert(sizeof(uint64_t) <= sizeof(unsigned long long), "sizeof(uint64_t) <= sizeof(long long) must be true");
        PyObject *val = PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(42));
        
        return val;
    }

    static PyMethodDef NNFCCodecState_methods[] = {
        {"start", (PyCFunction)NNFCCodecState_start, METH_VARARGS,
              "Starts the counter."
        },
        {"stop", (PyCFunction)NNFCCodecState_stop, METH_VARARGS,
              "Stops the counter."
        },
        {"reset", (PyCFunction)NNFCCodecState_reset, METH_VARARGS,
              "Resets the counter that the subsequent call to `start` will begin at zero."
        },
        {"getval", (PyCFunction)NNFCCodecState_getval, METH_VARARGS,
              "Returns current value of the counter."
        },
        {NULL, NULL, METH_VARARGS, ""}  /* Sentinel */
    };
    
    static PyTypeObject NNFCCodecStateType = {
        PyObject_HEAD_INIT(NULL)
        .tp_name = "nnfc_codec.NNFCCodecState",
        .tp_basicsize = sizeof(NNFCCodecState),
        .tp_itemsize = 0,
        .tp_dealloc = (destructor)NNFCCodecState_dealloc,
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
        .tp_doc = "NNFCCodecState object",
        .tp_traverse = 0,
        .tp_clear = 0,
        .tp_richcompare = 0,
        .tp_weaklistoffset = 0,
        .tp_iter = 0,
        .tp_iternext = 0,
        .tp_methods = NNFCCodecState_methods,
        .tp_members = 0,
        .tp_getset = 0,
        .tp_base = 0,
        .tp_dict = 0,
        .tp_descr_get = 0,
        .tp_descr_set = 0,
        .tp_dictoffset = 0,
        .tp_init = (initproc)NNFCCodecState_init,
        .tp_alloc = 0,
        .tp_new = NNFCCodecState_new,
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

    struct module_state {
        PyObject *error;
    };

    static PyObject* nnfc_codec_encode(PyObject *, PyObject *args) {

        PyObject *encoder_state;
        THFloatTensor *input;
        
        if (!PyArg_ParseTuple(args, "OO",
                              &encoder_state,
                              &input)){
            return 0;
        }

        PyObject *hello = PyUnicode_FromString("hello world");
        return hello;

    }

    static PyObject* nnfc_codec_decode(PyObject *args) {

        
        
        PyObject *hello = PyUnicode_FromString("hello world");
        return hello;
    }

    
    static PyMethodDef module_methods[] = {
        { "encode", (PyCFunction)nnfc_codec_encode, METH_VARARGS, "encode a batch of intermediate activations" },
        { "decode", (PyCFunction)nnfc_codec_decode, METH_VARARGS, "decode a batch of intermediate activations" },
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

        if (PyType_Ready(&NNFCCodecStateType) < 0){
            PyErr_SetString(PyExc_ValueError, "could not intialize NNFCCodecState object.");
            return 0;
        }

        PyObject* module = PyModule_Create(&module_def);

        if (module == NULL){
            PyErr_SetString(PyExc_ValueError, "could not create perlib module.");
            return 0;
        }

        Py_INCREF(&NNFCCodecStateType);
        PyModule_AddObject(module, "NNFCCodecState", (PyObject *)&NNFCCodecStateType);

        PyObject *module_namespace = PyModule_GetDict(module);
        PyObject *version = PyUnicode_FromString("0.0.0");

        if(PyDict_SetItemString(module_namespace, "__version__", version)){
            PyErr_SetString(PyExc_ValueError, "could not set __version__ of nnfc_codec module.");
            return 0;
        }

        return module;
    }

}
