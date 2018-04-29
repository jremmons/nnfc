#pragma once

#include <cstdint>
#include <vector>
#include "tensor.hh"

namespace nnfc {

    class SimpleEncoder
    {
    private:
        
    public:
        SimpleEncoder();
        ~SimpleEncoder();
        
        std::vector<uint8_t> encode(nn::Tensor<float, 3> input);
    };
    
    class SimpleDecoder
    {
    private:
        
    public:
        SimpleDecoder();
        ~SimpleDecoder();

        nn::Tensor<float, 3> decode(std::vector<uint8_t> input);
    };

}

