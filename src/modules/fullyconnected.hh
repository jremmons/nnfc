#ifndef _NN_FULLYCONNECTED
#define _NN_FULLYCONNECTED

#include "tensor.hh"

namespace NN {

    void fully_connected(const Tensor<float, 4> input,
                         const Tensor<float, 2> weights,
                         Tensor<float, 4> output);
    
}

#endif // _NN_FULLYCONNECTED
