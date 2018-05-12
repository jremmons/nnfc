#ifndef _NNFC_COMMON_H
#define _NNFC_COMMON_H

#include <Python.h>

#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "tensor.hh"
#include "nnfc_CXXAPI.hh"

#ifndef NODEBUG
#define _NNFC_DEBUG
#endif


class nnfc_python_exception : public std::exception {
private:
    std::shared_ptr<PyObject> error_type_;
    const std::string error_message_;

public:
    nnfc_python_exception(PyObject* error_type, const std::string error_message) :
        error_type_(error_type, [](PyObject*){}),
        error_message_(error_message)
    { }

    const char* what() const noexcept {
        return error_message_.c_str();
    }
    
    PyObject* type() const noexcept {
        return error_type_.get();
    }    
};


#ifdef _NNFC_DEBUG 
#define WrapperAssert(expr, error_type, error_message) NNFCPythonAssert((expr), error_type, error_message, __FILE__, __LINE__);
#else
#define WrapperAssert(expr, error_type, error_message) // assertions removed
#endif

inline void NNFCPythonAssert(bool expr,
                             PyObject* error_type,
                             const std::string error_message,
                             const char *file,
                             const long long int line)
{

    if(!expr){
        std::string augmented_message = error_message +
            std::string(" ") + 
            std::string(file) +
            std::string(":") +
            std::to_string(line);
        
        throw nnfc_python_exception(error_type, augmented_message);
    }

}

inline nnfc::cxxapi::constructor_list parse_dict(PyObject *args_dictionary,
                                                 nnfc::cxxapi::constructor_type_list args_list) {

    nnfc::cxxapi::constructor_list args({});
    const Py_ssize_t num_args = args_list.size();
    
    for(int i = 0; i < num_args; i++){
        const char* key = args_list[i].first.c_str();
        PyObject* val = PyDict_GetItemString(args_dictionary, key);

        WrapperAssert(val, PyExc_TypeError, args_list[i].first + " must be set in the args dictionary");
        
        auto type = args_list[i].second;
        if(typeid(int) == type) {
            WrapperAssert(PyLong_Check(val),
                          PyExc_ValueError,
                          args_list[i].first + " must be an integer.");

            int arg = PyLong_AsLong(val);
            args.push_back({std::string(key), arg});
        }
        else {    
            throw std::runtime_error("unknown type required by the context");
        }
    }

    return args;
}

#endif // _NNFC_COMMON_H
