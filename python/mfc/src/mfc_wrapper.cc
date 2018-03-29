#include <TH/TH.h>
#include <stdlib.h>

#include <cassert>
#include <cstdint>
#include <iostream>

#include "noop.hh"
#include "blob1d.hh"
#include "blob3d.hh"

// functions must 'extern "C"' in order to be callable from within pytorch/python
// https://github.com/torch/TH/blob/master/generic/THTensor.h

#define _TORCH_SUCCESS 1;
#define _TORCH_FAILURE 0;

#define DEBUG 1
#define ASSERT(expr) if(!(expr) and DEBUG){ std::cerr << "assertion failed at line: " <<  __LINE__ << " \n"; throw; } // TOOD(jremmons) make this more sensible...


class Blob1DTorchByte : public Blob1D<uint8_t> {
public:
    Blob1DTorchByte(uint8_t *data, size_t size, THByteTensor *tensor) :
        Blob1D<uint8_t>(data, size),
        tensor_(tensor)
    { }
    
    void resize(size_t new_size) {
        THByteTensor_resize1d(tensor_, new_size); // will realloc
        data_ = THByteTensor_data(tensor_);
        size_ = THByteTensor_size(tensor_, 0);
        
        ASSERT(size_ == new_size);
    }

private:
    THByteTensor *tensor_;
};


extern "C" int add_forward(THFloatTensor *input, THByteTensor *output)
{

    // sanity checking
    {
        int input_contiguous = THFloatTensor_isContiguous(input);
        ASSERT(input_contiguous);
        int output_contiguous = THByteTensor_isContiguous(output);
        ASSERT(output_contiguous);
        
        int input_ndims = THFloatTensor_nDimension(input);
        ASSERT(input_ndims == 3);
    }

    
    // munge the blobs
    size_t c_size = THFloatTensor_size(input, 0);
    size_t h_size = THFloatTensor_size(input, 1);
    size_t w_size = THFloatTensor_size(input, 2);

    float* input_data = THFloatTensor_data(input);
    Blob3D<float> input_blob{input_data, c_size, h_size, w_size};

    size_t b_size = 0;
    if(THByteTensor_nDimension(output) > 0){
        b_size = THByteTensor_size(output, 0);
    }
    uint8_t *output_data = THByteTensor_data(output);
    Blob1DTorchByte output_blob{output_data, b_size, output};
    

    // call the encoder
    Noop n;
    n.encode(input_blob, output_blob);

    return _TORCH_SUCCESS;    
}


extern "C" int add_backward(THFloatTensor *grad_output, THFloatTensor *grad_input)
{
    THFloatTensor_resizeAs(grad_input, grad_output);
    THFloatTensor_fill(grad_input, 1);

    return _TORCH_SUCCESS;
}
