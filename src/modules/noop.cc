#include <cassert>
#include <cstdint>
#include <iostream>

#include "noop.hh"
#include "blob1d.hh"
#include "blob4d.hh"

static uint64_t magic_num = 0xDEADBEEF;

// TODO(jremmons)
// put header information at the beginning of the blob
// add a get_size function
// put the file in protobuf format
// add a 'working memory' blob that can be pass into the function
// 

void Noop::encode(Blob4D<float> &input, Blob1D<uint8_t> &output) {

    output.resize(4 * input.batch_size * input.channels * input.height * input.width + 5*sizeof(uint64_t));
        
    // for(size_t n = 0; n < input.batch_size; n++){
    //     for(size_t i = 0; i < input.channels; i++){
    //         for(size_t j = 0; j < input.height; j++){
    //             for(size_t k = 0; k < input.width; k++){

    //                 float num = input.get(n, i ,j ,k); uint8_t
    //                 *bytes = reinterpret_cast<uint8_t*>(&num);

    //                 size_t offset = 4 * (input.channels*input.height*input.width*n + input.height*input.width*i + input.width*j + k);
    //                 output.set(bytes[0], offset + 0);
    //                 output.set(bytes[1], offset + 1);
    //                 output.set(bytes[2], offset + 2);
    //                 output.set(bytes[3], offset + 3);

    //             }
    //         }
    //     }
    // }

    std::memcpy(output.data, input.data, output.size - 5*sizeof(uint64_t));
    
    uint64_t batch_size = input.batch_size;
    uint64_t channels = input.channels;
    uint64_t height = input.height;
    uint64_t width = input.width;

    uint8_t *magic_num_bytes = reinterpret_cast<uint8_t*>(&magic_num);
    uint8_t *batch_size_bytes = reinterpret_cast<uint8_t*>(&batch_size);
    uint8_t *channels_bytes = reinterpret_cast<uint8_t*>(&channels);
    uint8_t *height_bytes = reinterpret_cast<uint8_t*>(&height);
    uint8_t *width_bytes = reinterpret_cast<uint8_t*>(&width);

    for(size_t i = 0; i < 8; i++){
        output.set(magic_num_bytes[i], output.size-8 + i);
        output.set(batch_size_bytes[i], output.size-16 + i);
        output.set(channels_bytes[i], output.size-24 + i);
        output.set(height_bytes[i], output.size-32 + i);
        output.set(width_bytes[i], output.size-40 + i);    
    }
    
}

void Noop::decode(Blob1D<uint8_t> &input, Blob4D<float> &output) {

    uint64_t magic_num_read;
    uint64_t batch_size;
    uint64_t channels;
    uint64_t height;
    uint64_t width;

    uint8_t *magic_num_bytes = reinterpret_cast<uint8_t*>(&magic_num_read);
    uint8_t *batch_size_bytes = reinterpret_cast<uint8_t*>(&batch_size);
    uint8_t *channels_bytes = reinterpret_cast<uint8_t*>(&channels);
    uint8_t *height_bytes = reinterpret_cast<uint8_t*>(&height);
    uint8_t *width_bytes = reinterpret_cast<uint8_t*>(&width);
    
    for(size_t i = 0; i < 8; i++){
        magic_num_bytes[i] = input.get(input.size-8 + i);
        batch_size_bytes[i] = input.get(input.size-16 + i);
        channels_bytes[i] = input.get(input.size-24 + i);
        height_bytes[i] = input.get(input.size-32 + i);
        width_bytes[i] = input.get(input.size-40 + i);    
    }

    assert(magic_num_read == magic_num);

    output.resize(batch_size, channels, height, width);
        
    // for(size_t n = 0; n < output.batch_size; n++){
    //     for(size_t i = 0; i < output.channels; i++){
    //         for(size_t j = 0; j < output.height; j++){
    //             for(size_t k = 0; k < output.width; k++){

    //                 float num;
    //                 uint8_t *bytes = reinterpret_cast<uint8_t*>(&num);

    //                 size_t offset = 4 * (output.channels*output.height*output.width*n + output.height*output.width*i + output.width*j + k);

    //                 bytes[0] = input.get(offset);
    //                 bytes[1] = input.get(offset + 1);
    //                 bytes[2] = input.get(offset + 2);
    //                 bytes[3] = input.get(offset + 3);

    //                 output.set(num, n, i, j, k);

    //             }
    //         }
    //     }
    // }

    std::memcpy(output.data, input.data, 4*output.size);
    
}
