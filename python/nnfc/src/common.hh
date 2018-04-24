#pragma once

#include <iostream>
#include <string>

inline void WrapperAssert(bool expr, const std::string message="unnammed runtime error"){
    #ifdef _NNFC_DEBUG 
    if(!expr){
        throw std::runtime_error(message);
    }
    #endif
}
