#pragma once

#include "array3d.hh"

class Noop {
public:
    Noop();
    ~Noop();
    
    Array3D<float> encode(Array3D<float> input);
    Array3D<float> decode(Array3D<float> input);
};
