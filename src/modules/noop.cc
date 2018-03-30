#include <cassert>
#include <cstdint>
#include <iostream>

#include "noop.hh"
#include "blob1d.hh"
#include "blob3d.hh"

Noop::Noop() {}
Noop::~Noop() {}

void Noop::encode(Blob3D<float> &input, Blob1D<uint8_t> &output) {
    output.resize(3*100*100 * 4);
        
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 100; j++){
            for(size_t k = 0; k < 100; k++){

                float num = input.get(i,j,k);
                uint8_t *bytes = reinterpret_cast<uint8_t*>(&num);
                
                output.set(bytes[0], 4*100*100*i + 4*100*j + 4*k + 0);
                output.set(bytes[1], 4*100*100*i + 4*100*j + 4*k + 1);
                output.set(bytes[2], 4*100*100*i + 4*100*j + 4*k + 2);
                output.set(bytes[3], 4*100*100*i + 4*100*j + 4*k + 3);

            }
        }
    }

}

void Noop::decode(Blob1D<uint8_t> &input, Blob3D<float> &output) {
    output.resize(3, 100, 100);
        
    for(size_t i = 0; i < 3; i++){
        for(size_t j = 0; j < 100; j++){
            for(size_t k = 0; k < 100; k++){
                
                float num;
                uint8_t *bytes = reinterpret_cast<uint8_t*>(&num);

                bytes[0] = input.get(4*100*100*i + 4*100*j + 4*k + 0);
                bytes[1] = input.get(4*100*100*i + 4*100*j + 4*k + 1);
                bytes[2] = input.get(4*100*100*i + 4*100*j + 4*k + 2);
                bytes[3] = input.get(4*100*100*i + 4*100*j + 4*k + 3);
                output.set(num, i, j, k);

            }
        }
    }

}
