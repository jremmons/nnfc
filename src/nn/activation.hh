#ifndef _NN_ACTIVATION_H
#define _NN_ACTIVATION_H

#include "tensor.hh"

namespace nn {

void relu(const Tensor<float, 4> input, Tensor<float, 4> output);
}

#endif  // _NN_ACTIVATION_H
