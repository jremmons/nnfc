#pragma once

#include "blob3d.hh"
#include "blob4d.hh"

namespace NN {

    void conv2d(const Blob4D<float> &input,
                const Blob4D<float> &kernel,
                Blob4D<float> &output,
                const size_t stride,
                const size_t zero_padding);
    
}
