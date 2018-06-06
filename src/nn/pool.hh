#ifndef _NN_POOL_H
#define _NN_POOL_H

#include "tensor.hh"

namespace nn {

void average_pooling(const Tensor<float, 4> input, Tensor<float, 4> output);
}

#endif  // _NN_POOL_H
