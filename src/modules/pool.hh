#pragma once

#include "tensor.hh"
#include "blob4d.hh"

namespace NN {

    void average_pooling(const Blob4D<float> &input,
                         Blob4D<float> &output);
    
    void average_pooling(const Tensor<float, 4> input,
                         Tensor<float, 4> output);

}
