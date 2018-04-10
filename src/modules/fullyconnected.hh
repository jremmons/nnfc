#pragma once

#include "blob2d.hh"
#include "blob4d.hh"

namespace NN {

    void fully_connected(const Blob4D<float> &input,
                         const Blob2D<float> &weights,
                         Blob4D<float> &output);
    
}
