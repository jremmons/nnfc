#include <cassert>
#include <cstdint>
#include <iostream>

#include "noop.hh"
#include "blob1d.hh"
#include "blob3d.hh"

Noop::Noop() {}
Noop::~Noop() {}

void Noop::encode(Blob3D<float> &input, Blob1D<uint8_t> &output) {
    output.resize(3*100*100);
        
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 100; j++){
            for(size_t k = 0; k < 100; k++){
                output.set(input.get(i,j,k), 100*100*i + 100*j + k);
            }
        }
    }
}

void Noop::decode(Blob1D<uint8_t> &input, Blob3D<float> &output) {
    output.resize(3, 100, 100);
        
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 100; j++){
            for(size_t k = 0; k < 100; k++){
                output.set(input.get(100*100*i + 100*j + k), i, j, k);
            }
        }
    }
}
