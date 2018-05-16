#include <turbojpeg.h>

#include <chrono>
#include <cassert>
#include <cstdint>
#include <vector>
#include <iostream>
#include <cmath>

#include "tensor.hh"
#include "jpeg_codec.hh"

nnfc::JPEGEncoder::JPEGEncoder(int quantizer) :
    quantizer_(quantizer),
    jpeg_compressor(tjInitCompress(), [](void* ptr){ tjDestroy(ptr); })
{ }

nnfc::JPEGEncoder::~JPEGEncoder()
{ }

std::vector<uint8_t> nnfc::JPEGEncoder::forward(nn::Tensor<float, 3> input)
{
    auto total_t1 = std::chrono::high_resolution_clock::now();

    uint64_t dim0 = input.dimension(0);
    uint64_t dim1 = input.dimension(1);
    uint64_t dim2 = input.dimension(2);

    auto min_max_t1 = std::chrono::high_resolution_clock::now();
    float min = input(0,0,0);
    float max = input(0,0,0);
    for(size_t channel = 0; channel < dim0; channel++) {
        for(size_t row = 0; row < dim1; row++) {
            for(size_t col = 0; col < dim2; col++) {
                
                float val = input(channel, row, col);

                if(val > max) {
                    max = val;
                }
                if(val < min) {
                    min = val;
                }
                
            }
        }
    }
    auto min_max_t2 = std::chrono::high_resolution_clock::now();
    //std::cout << min << " " << max << "\n";    
    
    // create a square grid for the activations to go into
    const size_t jpeg_chunks = std::ceil(std::sqrt(dim0));
    const size_t jpeg_height = jpeg_chunks*dim1;
    const size_t jpeg_width = jpeg_chunks*dim2;

    auto buffer_allocate_t1 = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> buffer(jpeg_height * jpeg_width+1024);
    std::fill(buffer.begin(), buffer.end(), 0);    
    auto buffer_allocate_t2 = std::chrono::high_resolution_clock::now();

    // compute the strides for laying out the data in memory
    const size_t row_channel_stride = dim1 * dim2;
    const size_t row_stride = jpeg_chunks * dim2;
    const size_t channel_stride = dim2;
    const size_t col_stride = 1;

    // swizzle the data into the right memory layout
    auto swizzle_t1 = std::chrono::high_resolution_clock::now();
    for(size_t row_channel = 0; row_channel < jpeg_chunks*jpeg_chunks; row_channel += jpeg_chunks) {

        for(size_t row = 0; row < dim1; row++) {
            for(size_t channel = 0; channel < jpeg_chunks; channel++) {

                if(row_channel + channel < dim0) {
                    for(size_t col = 0; col < dim2; col++) {
                        
                        const float val = input(row_channel + channel, row, col);
                        const size_t offset = row_channel_stride*row_channel + 
                                              row_stride*row +
                                              channel_stride*channel +
                                              col_stride*col;

                        buffer[offset] = static_cast<uint8_t>((val - min) * (255 / (max - min)));
                        
                    }
                }
            }
        }
    }
    auto swizzle_t2 = std::chrono::high_resolution_clock::now();

    // jpeg compress the data
    auto compress_t1 = std::chrono::high_resolution_clock::now();
    long unsigned int jpeg_size = 0;
    unsigned char* compressed_image = nullptr;
    tjCompress2(jpeg_compressor.get(), buffer.data(), jpeg_width, 0, jpeg_height, TJPF_GRAY,
                &compressed_image, &jpeg_size, TJSAMP_GRAY, quantizer_, TJFLAG_FASTDCT);
    auto compress_t2 = std::chrono::high_resolution_clock::now();

    
    auto serialize_t1 = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> encoding(jpeg_size);
    std::memcpy(encoding.data(), compressed_image, jpeg_size);
    tjFree(compressed_image);    
    
    uint8_t *min_bytes = reinterpret_cast<uint8_t*>(&min);
    uint8_t *max_bytes = reinterpret_cast<uint8_t*>(&max);
    for(size_t i = 0; i < sizeof(float); i++){
        encoding.push_back(min_bytes[i]);
    }
    for(size_t i = 0; i < sizeof(float); i++){
        encoding.push_back(max_bytes[i]);
    }
    
    uint8_t *dim0_bytes = reinterpret_cast<uint8_t*>(&dim0);
    uint8_t *dim1_bytes = reinterpret_cast<uint8_t*>(&dim1);
    uint8_t *dim2_bytes = reinterpret_cast<uint8_t*>(&dim2);
    for(size_t i = 0; i < sizeof(uint64_t); i++){
        encoding.push_back(dim0_bytes[i]);
    }
    for(size_t i = 0; i < sizeof(uint64_t); i++){
        encoding.push_back(dim1_bytes[i]);
    }
    for(size_t i = 0; i < sizeof(uint64_t); i++){
        encoding.push_back(dim2_bytes[i]);
    }
    auto serialize_t2 = std::chrono::high_resolution_clock::now();

    auto total_t2 = std::chrono::high_resolution_clock::now();
    const double total_time = std::chrono::duration_cast<std::chrono::duration<double>>(total_t2 - total_t1).count();

    std::cout << "min_max:\t" << std::chrono::duration_cast<std::chrono::duration<double>>(min_max_t2 - min_max_t1).count() << "\t" <<
        std::chrono::duration_cast<std::chrono::duration<double>>(min_max_t2 - min_max_t1).count() / total_time << "\n";
    std::cout << "buffer_alloc:\t" << std::chrono::duration_cast<std::chrono::duration<double>>(buffer_allocate_t2 - buffer_allocate_t1).count() << "\t" <<
        std::chrono::duration_cast<std::chrono::duration<double>>(buffer_allocate_t2 - buffer_allocate_t1).count() / total_time << "\n";
    std::cout << "swizzle:\t" << std::chrono::duration_cast<std::chrono::duration<double>>(swizzle_t2 - swizzle_t1).count() << "\t" <<
        std::chrono::duration_cast<std::chrono::duration<double>>(swizzle_t2 - swizzle_t1).count() / total_time << "\n";
    std::cout << "compress:\t" << std::chrono::duration_cast<std::chrono::duration<double>>(compress_t2 - compress_t1).count() << "\t" <<
        std::chrono::duration_cast<std::chrono::duration<double>>(compress_t2 - compress_t1).count() / total_time << "\n";
    std::cout << "serialize:\t" << std::chrono::duration_cast<std::chrono::duration<double>>(serialize_t2 - serialize_t1).count() << "\t" <<
        std::chrono::duration_cast<std::chrono::duration<double>>(serialize_t2 - serialize_t1).count() / total_time << "\n";
    std::cout << ">>>total:\t" << total_time << "\n";

    return encoding;    
}

nn::Tensor<float, 3> nnfc::JPEGEncoder::backward(nn::Tensor<float, 3> input)
{
    return input;
}

nnfc::JPEGDecoder::JPEGDecoder() :
    jpeg_decompressor(tjInitDecompress(), [](void* ptr){ tjDestroy(ptr); })
{ }

nnfc::JPEGDecoder::~JPEGDecoder()
{ }

nn::Tensor<float, 3> nnfc::JPEGDecoder::forward(std::vector<uint8_t> input)
{

    uint64_t dim0;
    uint64_t dim1;
    uint64_t dim2;
    uint8_t *dim0_bytes = reinterpret_cast<uint8_t*>(&dim0);
    uint8_t *dim1_bytes = reinterpret_cast<uint8_t*>(&dim1);
    uint8_t *dim2_bytes = reinterpret_cast<uint8_t*>(&dim2);

    size_t length = input.size();
    size_t dim0_offset = length - 3*sizeof(uint64_t);
    size_t dim1_offset = length - 2*sizeof(uint64_t);
    size_t dim2_offset = length - 1*sizeof(uint64_t);
    for(size_t i = 0; i < sizeof(uint64_t); i++){
        dim0_bytes[i] = input[i + dim0_offset];
        dim1_bytes[i] = input[i + dim1_offset];
        dim2_bytes[i] = input[i + dim2_offset];
    }

    float min;
    float max;
    uint8_t *min_bytes = reinterpret_cast<uint8_t*>(&min);
    uint8_t *max_bytes = reinterpret_cast<uint8_t*>(&max);
    size_t min_offset = length - 3*sizeof(uint64_t) - 2*sizeof(float);
    size_t max_offset = length - 3*sizeof(uint64_t) - 1*sizeof(float);
    for(size_t i = 0; i < sizeof(uint64_t); i++){
        min_bytes[i] = input[i + min_offset];
        max_bytes[i] = input[i + max_offset];
    }    
    std::cout << min << " " << max << "\n";
    
    const size_t jpeg_chunks = std::ceil(std::sqrt(dim0));
    // const size_t jpeg_height = jpeg_chunks*dim1;
    // const size_t jpeg_width = jpeg_chunks*dim2;

    const long unsigned int jpeg_size = input.size() - 3*sizeof(uint64_t) - 2*sizeof(float);

    int jpegSubsamp, width, height;
    tjDecompressHeader2(jpeg_decompressor.get(), input.data(), jpeg_size, &width, &height, &jpegSubsamp);

    // std::cout << jpeg_height << " " << height << "\n";
    // std::cout << jpeg_width << " " << width << "\n";
    
    std::vector<uint8_t> buffer(width * height);
    
    tjDecompress2(jpeg_decompressor.get(), input.data(), jpeg_size, buffer.data(), width, 0/*pitch*/, height, TJPF_GRAY, TJFLAG_FASTDCT);
    
    // compute the strides for laying out the data in memory
    const size_t row_channel_stride = dim1 * dim2;
    const size_t row_stride = jpeg_chunks * dim2;
    const size_t channel_stride = dim2;
    const size_t col_stride = 1;

    // swizzle the data into the right memory layout
    nn::Tensor<float, 3> output(dim0, dim1, dim2);
        
    for(size_t row_channel = 0; row_channel < jpeg_chunks*jpeg_chunks; row_channel += jpeg_chunks) {

        for(size_t row = 0; row < dim1; row++) {
            for(size_t channel = 0; channel < jpeg_chunks; channel++) {

                if(row_channel + channel < dim0) {
                    for(size_t col = 0; col < dim2; col++) {
                        
                        const size_t offset = row_channel_stride*row_channel + 
                                              row_stride*row +
                                              channel_stride*channel +
                                              col_stride*col;
                        const double val = buffer[offset]; 
                        output(row_channel + channel, row, col) = static_cast<float>((max-min) * (val/255) + min);
                    }
                }
            }
        }
    }  
    
    return output;
}

nn::Tensor<float, 3> nnfc::JPEGDecoder::backward(nn::Tensor<float, 3> input)
{
    return input;
}
