#pragma once

#include <cstdint>

#include "blob1d.hh"
#include "blob4d.hh"

namespace Noop {

    void encode(Blob4D<float> &input, Blob1D<uint8_t> &output);
    void decode(Blob1D<uint8_t> &input, Blob4D<float> &output);

}
