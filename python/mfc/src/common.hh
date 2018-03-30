#pragma once

#include <iostream>

#define DEBUG 1
#define ASSERT(expr) if(!(expr) and DEBUG){ std::cerr << "assertion failed at line: " <<  __LINE__ << " \n"; throw; } // TOOD(jremmons) make this more sensible...

#define _TORCH_SUCCESS 1;
#define _TORCH_FAILURE 0;

