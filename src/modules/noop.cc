#include <cassert>
#include <cstdint>
#include <iostream>

#include "noop.hh"
#include "blob1d.hh"
#include "blob3d.hh"

static uint64_t magic_num = 0xDEADBEEF;

Noop::Noop() {}
Noop::~Noop() {}

void Noop::encode(Blob3D<float> &input, Blob1D<uint8_t> &output) {

    output.resize(4 * input.channels * input.height * input.width + 4*8);
        
    for(size_t i = 0; i < input.channels; i++){
        for(size_t j = 0; j < input.height; j++){
            for(size_t k = 0; k < input.width; k++){

                float num = input.get(i,j,k);
                uint8_t *bytes = reinterpret_cast<uint8_t*>(&num);
                
                output.set(bytes[0], 4*input.height*input.width*i + 4*input.width*j + 4*k + 0);
                output.set(bytes[1], 4*input.height*input.width*i + 4*input.width*j + 4*k + 1);
                output.set(bytes[2], 4*input.height*input.width*i + 4*input.width*j + 4*k + 2);
                output.set(bytes[3], 4*input.height*input.width*i + 4*input.width*j + 4*k + 3);

            }
        }
    }

    uint64_t channels = input.channels;
    uint64_t height = input.height;
    uint64_t width = input.width;

    uint8_t *magic_num_bytes = reinterpret_cast<uint8_t*>(&magic_num);
    uint8_t *channels_bytes = reinterpret_cast<uint8_t*>(&channels);
    uint8_t *height_bytes = reinterpret_cast<uint8_t*>(&height);
    uint8_t *width_bytes = reinterpret_cast<uint8_t*>(&width);

    for(size_t i = 0; i < 8; i++){
        output.set(magic_num_bytes[i], output.size-8 + i);
        output.set(channels_bytes[i], output.size-16 + i);
        output.set(height_bytes[i], output.size-24 + i);
        output.set(width_bytes[i], output.size-32 + i);    
    }
    
}

void Noop::decode(Blob1D<uint8_t> &input, Blob3D<float> &output) {

    uint64_t magic_num_read;
    uint64_t channels;
    uint64_t height;
    uint64_t width;

    uint8_t *magic_num_bytes = reinterpret_cast<uint8_t*>(&magic_num_read);
    uint8_t *channels_bytes = reinterpret_cast<uint8_t*>(&channels);
    uint8_t *height_bytes = reinterpret_cast<uint8_t*>(&height);
    uint8_t *width_bytes = reinterpret_cast<uint8_t*>(&width);
    
    for(size_t i = 0; i < 8; i++){
        magic_num_bytes[i] = input.get(input.size-8 + i);
        channels_bytes[i] = input.get(input.size-16 + i);
        height_bytes[i] = input.get(input.size-24 + i);
        width_bytes[i] = input.get(input.size-32 + i);    
    }

    assert(magic_num_read == magic_num);

    output.resize(channels, height, width);
        
    for(size_t i = 0; i < output.channels; i++){
        for(size_t j = 0; j < output.height; j++){
            for(size_t k = 0; k < output.width; k++){

                float num;
                uint8_t *bytes = reinterpret_cast<uint8_t*>(&num);
                bytes[0] = input.get(4*output.height*output.width*i + 4*output.width*j + 4*k);
                bytes[1] = input.get(4*output.height*output.width*i + 4*output.width*j + 4*k + 1);
                bytes[2] = input.get(4*output.height*output.width*i + 4*output.width*j + 4*k + 2);
                bytes[3] = input.get(4*output.height*output.width*i + 4*output.width*j + 4*k + 3);
                
                output.set(num, i, j, k);

            }
        }
    }


    // for(size_t i = 0; i < input.size; i+=4){

    //     float num;
    //     uint8_t *bytes = reinterpret_cast<uint8_t*>(&num);
        
    //     bytes[0] = input.get(i + 0);
    //     bytes[1] = input.get(i + 1);
    //     bytes[2] = input.get(i + 2);
    //     bytes[3] = input.get(i + 3);
    //     output.set(num, i/4, 0, 0);
        
    // }

}
