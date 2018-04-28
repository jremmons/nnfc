#ifndef _NN_POOL
#define _NN_POOL

#include "tensor.hh"

namespace NN {

    void average_pooling(const Tensor<float, 4> input,
                         Tensor<float, 4> output);

}

#endif // _NN_POOL
