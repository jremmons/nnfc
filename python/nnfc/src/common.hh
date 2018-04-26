#ifndef _NNFC_COMMON
#define _NNFC_COMMON

extern "C" {
#include <Python.h>
#include <numpy/arrayobject.h>
}

#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "tensor.hh"

#ifndef NODEBUG
#define _NNFC_DEBUG
#endif

class nnfc_python_exception : public std::exception {
private:
    PyObject* error_type_;
    const std::string error_message_;

public:
    nnfc_python_exception(PyObject* error_type, const std::string error_message) :
        error_type_(error_type),
        error_message_(error_message)
    { }

    const char* what() const noexcept {
        return error_message_.c_str();
    }
    
    PyObject* type() const noexcept {
        return error_type_;
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
        std::string augmented_message = error_message + std::string(" ") + 
            std::string(file) + std::string(":") + std::to_string(line);
            
        throw nnfc_python_exception(error_type, augmented_message);
    }

}

#endif // _NNFC_COMMON
