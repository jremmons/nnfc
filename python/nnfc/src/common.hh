#pragma once

#include <iostream>

#define DEBUG 1
#define ASSERT(expr) if(!(expr) and DEBUG){ std::cerr << "assertion failed: " << __FILE__ << ":" <<  __LINE__ << " \n"; throw; } // TOOD(jremmons) make this more sensible...

inline void WrapperAssert(bool expr, const char* message="unnammed runtime error"){
    if(!expr){
        throw std::runtime_error(message);
    }
}

#define _TORCH_SUCCESS 1;
#define _TORCH_FAILURE 0;

