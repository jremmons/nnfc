#pragma once

#include <iostream>
#include <string>

#define _DEBUG 1

inline void WrapperAssert(bool expr, const std::string message="unnammed runtime error"){
    #ifdef _DEBUG 
    if(!expr){
        throw std::runtime_error(message);
    }
    #endif
}

#define _TORCH_SUCCESS 1;
#define _TORCH_FAILURE 0;

