#ifndef _NNFC_JPEG_IMAGE_H
#define _NNFC_JPEG_IMAGE_H

#include <cstdint>
#include <functional>
#include <vector>

extern "C" {
#include <jpeglib.h>
}

#include <turbojpeg.h>

#include "codec/jpeg.hh"
#include "nn/tensor.hh"
#include "nnfc_CXXAPI.hh"

namespace nnfc {

class JPEGImageEncoder {
 private:
  codec::JPEGEncoder encoder_;

 public:
  JPEGImageEncoder(int quality);
  ~JPEGImageEncoder() {}

  std::vector<uint8_t> forward(nn::Tensor<float, 3> input);
  nn::Tensor<float, 3> backward(nn::Tensor<float, 3> input);

  static nnfc::cxxapi::constructor_type_list initialization_params() {
    return {{"quantizer", typeid(int)}};
  }
};

class JPEGImageDecoder {
 private:
  std::unique_ptr<void, void (*)(void*)> jpeg_decompressor;

 public:
  JPEGImageDecoder();
  ~JPEGImageDecoder();

  nn::Tensor<float, 3> forward(std::vector<uint8_t> input);
  nn::Tensor<float, 3> backward(nn::Tensor<float, 3> input);

  static nnfc::cxxapi::constructor_type_list initialization_params() {
    return {};
  }
};
}  // namespace nnfc

#endif  // _NNFC_JPEG_IMAGE_H
