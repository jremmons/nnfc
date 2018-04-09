#pragma once

#include "blob1d.hh"
#include "blob4d.hh"

namespace NN {

    void relu(const Blob4D<float> &input, Blob4D<float> &output);
    
}
