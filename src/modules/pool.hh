#pragma once

#include "blob4d.hh"

namespace NN {

    void average_pooling(const Blob4D<float> &input,
                         Blob4D<float> &output);
    
}
