#include <cassert>
#include <cstdint>
#include <iostream>

#include <turbojpeg.h>

#include "nnfc.hh"
#include "blob1d.hh"
#include "blob4d.hh"

static uint64_t magic_num = 0xDEADBEEF;

// TODO(jremmons)
// put header information at the beginning of the blob
// add a get_size function
// put the file in protobuf format
// add a 'working memory' blob that can be pass into the function
// 

void NNFC::encode(Blob4D<float> &input, Blob1D<uint8_t> &output) {

    std::cerr << "nnfc encoder was called!" << std::endl;

    static_assert(sizeof(double) == sizeof(uint64_t), "the current code assumes doubles are 64-bits long");
    size_t metadata_length = 6*sizeof(uint64_t);
    output.resize(sizeof(uint8_t) * input.batch_size * input.channels * input.height * input.width + metadata_length);
    
    float max_valf = 0.0;
    float min_valf = 0.0;

    // get the min and the max values
    for(size_t n = 0; n < input.batch_size; n++){
        for(size_t i = 0; i < input.channels; i++){
            for(size_t j = 0; j < input.height; j++){
                for(size_t k = 0; k < input.width; k++){

                    float num = input.get(n, i ,j ,k); 
                    if(num > max_valf){
                        max_valf = num;
                    }
                    if(num < min_valf){
                        min_valf = num;
                    }

                }
            }
        }
    }
    
    assert(max_valf > 0);
    assert(min_valf == 0);
    std::cerr << "min_val: " << min_valf  << " max_val: " << max_valf << std::endl;

    tjhandle _jpegCompressor = tjInitCompress();
    tjDestroy(_jpegCompressor);
    
    for(size_t n = 0; n < input.batch_size; n++){
        for(size_t i = 0; i < input.channels; i++){
            for(size_t j = 0; j < input.height; j++){
                for(size_t k = 0; k < input.width; k++){

                    float num = input.get(n, i ,j ,k); 

                    num = num / (max_valf/255.0); // squash between 0 and 255.0
                    uint8_t quantized_val = static_cast<uint8_t>(num);
                    
                    size_t offset = input.channels*input.height*input.width*n + input.height*input.width*i + input.width*j + k;
                    output.set(quantized_val, offset + 0);

                }
            }
        }
    }
    
    uint64_t batch_size = input.batch_size;
    uint64_t channels = input.channels;
    uint64_t height = input.height;
    uint64_t width = input.width;
    double max_val = max_valf;
    
    uint8_t *magic_num_bytes = reinterpret_cast<uint8_t*>(&magic_num);
    uint8_t *batch_size_bytes = reinterpret_cast<uint8_t*>(&batch_size);
    uint8_t *channels_bytes = reinterpret_cast<uint8_t*>(&channels);
    uint8_t *height_bytes = reinterpret_cast<uint8_t*>(&height);
    uint8_t *width_bytes = reinterpret_cast<uint8_t*>(&width);
    uint8_t *max_val_bytes = reinterpret_cast<uint8_t*>(&max_val);
    
    for(size_t i = 0; i < 8; i++){
        output.set(magic_num_bytes[i], output.size-8 + i);
        output.set(batch_size_bytes[i], output.size-16 + i);
        output.set(channels_bytes[i], output.size-24 + i);
        output.set(height_bytes[i], output.size-32 + i);
        output.set(width_bytes[i], output.size-40 + i);    
        output.set(max_val_bytes[i], output.size-48 + i);            
    }
    
}

void NNFC::decode(Blob1D<uint8_t> &input, Blob4D<float> &output) {

    std::cerr << "nnfc decoder was called!" << std::endl;

    uint64_t magic_num_read;
    uint64_t batch_size;
    uint64_t channels;
    uint64_t height;
    uint64_t width;
    double max_val;

    uint8_t *magic_num_bytes = reinterpret_cast<uint8_t*>(&magic_num_read);
    uint8_t *batch_size_bytes = reinterpret_cast<uint8_t*>(&batch_size);
    uint8_t *channels_bytes = reinterpret_cast<uint8_t*>(&channels);
    uint8_t *height_bytes = reinterpret_cast<uint8_t*>(&height);
    uint8_t *width_bytes = reinterpret_cast<uint8_t*>(&width);
    uint8_t *max_val_bytes = reinterpret_cast<uint8_t*>(&max_val);
    
    for(size_t i = 0; i < 8; i++){
        magic_num_bytes[i] = input.get(input.size-8 + i);
        batch_size_bytes[i] = input.get(input.size-16 + i);
        channels_bytes[i] = input.get(input.size-24 + i);
        height_bytes[i] = input.get(input.size-32 + i);
        width_bytes[i] = input.get(input.size-40 + i);    
        max_val_bytes[i] = input.get(input.size-48 + i);    
    }

    assert(magic_num_read == magic_num);

    output.resize(batch_size, channels, height, width);
        
    for(size_t n = 0; n < output.batch_size; n++){
        for(size_t i = 0; i < output.channels; i++){
            for(size_t j = 0; j < output.height; j++){
                for(size_t k = 0; k < output.width; k++){

                    size_t offset = output.channels*output.height*output.width*n + output.height*output.width*i + output.width*j + k;
                    uint8_t quantized_val = input.get(offset);
                    
                    float uncompressed_val = (max_val * static_cast<double>(quantized_val)) / 255.0;

                    output.set(uncompressed_val, n, i, j, k);

                }
            }
        }
    }

}
