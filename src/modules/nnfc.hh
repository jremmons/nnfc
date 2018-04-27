#pragma once

#include <cstdint>
#include <vector>
#include "tensor.hh"

namespace NNFC {

    class SimpleEncoder
    {
    private:
        
    public:
        SimpleEncoder();
        ~SimpleEncoder();
        
        std::vector<uint8_t> encode(NN::Tensor<float, 3> input);
    };
    
    class SimpleDecoder
    {
    private:
        
    public:
        SimpleDecoder();
        ~SimpleDecoder();

        NN::Tensor<float, 3> decode(std::vector<uint8_t> input);
    };

}

