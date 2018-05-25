#ifndef _NNFC_JPEG_H
#define _NNFC_JPEG_H

#include <cstdint>
#include <vector>
#include <functional>

extern "C" {
  #include <jpeglib.h>
}

#include <turbojpeg.h>

#include "nnfc_CXXAPI.hh"
#include "tensor.hh"

namespace nnfc {

    class JPEGEncoder
    {
    private:
        const int quality_;
        std::vector<uint8_t> compress(std::vector<uint8_t> & buffer,
                                      const size_t width, const size_t height);

    public:
        JPEGEncoder(int quality);
        ~JPEGEncoder() {}

        std::vector<uint8_t> forward(nn::Tensor<float, 3> input);
        nn::Tensor<float, 3> backward(nn::Tensor<float, 3> input);

        static nnfc::cxxapi::constructor_type_list initialization_params() { return { {"quantizer", typeid(int)} }; }
    };

    class JPEGDecoder
    {
    private:
        std::unique_ptr<void, void(*)(void*)> jpeg_decompressor;

    public:
        JPEGDecoder();
        ~JPEGDecoder();

        nn::Tensor<float, 3> forward(std::vector<uint8_t> input);
        nn::Tensor<float, 3> backward(nn::Tensor<float, 3> input);

        static nnfc::cxxapi::constructor_type_list initialization_params() { return {}; }
    };

}

#endif // _NNFC_JPEG_H
