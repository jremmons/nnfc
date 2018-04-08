#pragma once

#include "blob1d.hh"
#include "blob4d.hh"

namespace NN {

    void conv2d(Blob4D<float> &input, Blob1D<uint64_t> &labels);
    
}
