#ifndef _NN_NORMALIZATION_H
#define _NN_NORMALIZATION_H

#include "tensor.hh"

namespace nn {

void batch_norm(const Tensor<float, 4> input, const Tensor<float, 1> means,
                const Tensor<float, 1> variances, const Tensor<float, 1> weight,
                const Tensor<float, 1> bias, Tensor<float, 4> output,
                const float eps = 0.00001);
}

#endif  // _NN_NORMALIZATION_H
