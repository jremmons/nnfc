#include <cassert>
#include <cstdint>
#include <iostream>

#include "noop.hh"
#include "blob1d.hh"
#include "blob3d.hh"

Noop::Noop() {}
Noop::~Noop() {}

void Noop::encode(Blob3D<float> &input, Blob1D<uint8_t> &output) {
    input.set(0,0,0,0);

    size_t size = 256;
    output.resize(size);
    for(size_t i = 0; i < size; i++){
        output.set(i, i);
    }
}

void Noop::decode(Blob1D<uint8_t> &input, Blob3D<float> &output) {
    input.set(0,0);
    output.set(0,0,0,0);
}
