#ifndef _NN_FULLYCONNECTED_H
#define _NN_FULLYCONNECTED_H

#include "tensor.hh"

namespace nn {

    void fully_connected(const Tensor<float, 4> input,
                         const Tensor<float, 2> weights,
                         Tensor<float, 4> output);

    void fully_connected_with_bias(const Tensor<float, 4> input,
                                   const Tensor<float, 2> weights,
                                   const Tensor<float, 1> bias,
                                   Tensor<float, 4> output);
    
}

#endif // _NN_FULLYCONNECTED_H
