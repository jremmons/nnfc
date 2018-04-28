#pragma once

#include "tensor.hh"
#include "blob2d.hh"
#include "blob4d.hh"

namespace NN {

    void fully_connected(const Blob4D<float> &input,
                         const Blob2D<float> &weights,
                         Blob4D<float> &output);

    void fully_connected(const Tensor<float, 4> input,
                         const Tensor<float, 2> weights,
                         Tensor<float, 4> output);
    
}
