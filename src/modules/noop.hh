#pragma once

#include <cstdint>

#include "blob1d.hh"
#include "blob3d.hh"

class Noop {
public:
    Noop();
    ~Noop();
    
    void encode(Blob3D<float> &input, Blob1D<uint8_t> &output);
    void decode(Blob1D<uint8_t> &input, Blob3D<float> &output);
};
