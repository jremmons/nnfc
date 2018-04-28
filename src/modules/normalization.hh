#pragma once

#include "tensor.hh"
#include "blob1d.hh"
#include "blob4d.hh"

namespace NN {

    void batch_norm(const Blob4D<float> &input, 
                    const Blob1D<float> &means,
                    const Blob1D<float> &variances,
                    const Blob1D<float> &weight,
                    const Blob1D<float> &bias,
                    Blob4D<float> &output,
                    const float eps = 0.00001);
    
    void batch_norm(const Tensor<float, 4> input, 
                    const Tensor<float, 1> means,
                    const Tensor<float, 1> variances,
                    const Tensor<float, 1> weight,
                    const Tensor<float, 1> bias,
                    Tensor<float, 4> output,
                    const float eps = 0.00001);

}
