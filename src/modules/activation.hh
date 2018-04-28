#ifndef _NN_ACTIVATION
#define _NN_ACTIVATION

#include "tensor.hh"

namespace NN {

    void relu(const Tensor<float, 4> input, Tensor<float, 4> output);
    
}

#endif // _NN_ACTIVATION
