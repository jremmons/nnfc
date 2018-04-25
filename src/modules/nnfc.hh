#pragma once

#include <cstdint>
#include <vector>
#include <Eigen/CXX11/Tensor>

#include "blob.hh"
#include "blob1d.hh"
#include "blob4d.hh"

namespace NNFC {
    void encode(const Blob<float, 4> &input_blob, Blob<uint8_t, 1> &output);
    void decode(const Blob<uint8_t, 1> &input_blob, Blob<float, 4> &output_blob);

    // class NoopEncoder
    // {
    // private:
        
    // public:
    //     NoopEncoder();
    //     ~NoopEncoder();

    //     std::vector<uint8_t> encode(input_blob, output_buffer=std::vector<uint8_t>());
    // };
    
    // class NoopDecoder
    // {
    // private:
        
    // public:
    //     NoopDecoder();
    //     ~NoopDecoder();

    //     output_blob decode(std::vector<uint8_t> compressed_blob, output_buffer);
    // };

}

