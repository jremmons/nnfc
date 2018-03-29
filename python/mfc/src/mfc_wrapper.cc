#include <TH/TH.h>

#include <iostream>

#include "noop.hh"
#include "array3d.hh"

// functions must 'extern "C"' in order to be callable from within pytorch/python
// https://github.com/torch/TH/blob/master/generic/THTensor.h

#define _TORCH_SUCCESS 1;
#define _TORCH_FAILURE 0;

#define DEBUG 1
#define ASSERT(expr) if(!(expr) and DEBUG){ std::cerr << "assertion failed at line: " <<  __LINE__ << " \n"; throw; } // TOOD(jremmons) make this more sensible...

extern "C" int add_forward(THFloatTensor *input, THFloatTensor *output)
{

    // sanity checking
    {
        THFloatTensor_resizeAs(output, input);
        
        int same_size = THFloatTensor_isSameSizeAs(input, output);
        ASSERT(same_size);
        
        int input_contiguous = THFloatTensor_isContiguous(input);
        ASSERT(input_contiguous);
        int output_contiguous = THFloatTensor_isContiguous(input);
        ASSERT(output_contiguous);
        
        int ndims = THFloatTensor_nDimension(input);
        ASSERT(ndims == 3);
    }
    
    size_t c_size = THFloatTensor_size(input, 0);
    size_t h_size = THFloatTensor_size(input, 1);
    size_t w_size = THFloatTensor_size(input, 2);
    
    float* input_data = THFloatTensor_data(input);
    Array3D<float> input_blob{input_data, c_size, h_size, w_size};
    Noop n;

    Array3D<float> output_blob = n.encode(input_blob);

    return _TORCH_SUCCESS;    
}

extern "C" int add_backward(THFloatTensor *grad_output, THFloatTensor *grad_input)
{
    THFloatTensor_resizeAs(grad_input, grad_output);
    THFloatTensor_fill(grad_input, 1);

    return _TORCH_SUCCESS;
}
