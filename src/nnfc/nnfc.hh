#ifndef _NNFC_NNFC_H
#define _NNFC_NNFC_H

#include <cstdint>
#include <vector>

#include "nnfc_CXXAPI.hh"
#include "tensor.hh"

namespace nnfc {

    class SimpleEncoder
    {
    private:
        
    public:
        SimpleEncoder();
        ~SimpleEncoder();

        std::vector<uint8_t> forward(nn::Tensor<float, 3> input);
        nn::Tensor<float, 3> backward(nn::Tensor<float, 3> input);

        static nnfc::cxxapi::constructor_type_list initialization_params() { return {}; }
    };
    
    class SimpleDecoder
    {
    private:
        
    public:
        SimpleDecoder();
        ~SimpleDecoder();

        nn::Tensor<float, 3> forward(std::vector<uint8_t> input);
        nn::Tensor<float, 3> backward(nn::Tensor<float, 3> input);

        static nnfc::cxxapi::constructor_type_list initialization_params() { return {}; }
    };

}

#endif // _NNFC_NNFC_H
