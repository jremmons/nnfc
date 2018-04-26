#pragma once

#include <cstdint>
#include <vector>
#include "tensor.hh"

namespace NNFC {
    // void encode(const Blob<float, 4> &input_blob, Blob<uint8_t, 1> &output);
    // void decode(const Blob<uint8_t, 1> &input_blob, Blob<float, 4> &output_blob);

    class SimpleEncoder
    {
    private:
        
    public:
        SimpleEncoder();
        ~SimpleEncoder();
        
        std::vector<uint8_t> encode(NNFC::Tensor<float, 3> input);
    };
    
    class SimpleDecoder
    {
    private:
        
    public:
        SimpleDecoder();
        ~SimpleDecoder();

        NNFC::Tensor<float, 3> decode(std::vector<uint8_t> input);
    };

}

