#pragma once

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
    
}
