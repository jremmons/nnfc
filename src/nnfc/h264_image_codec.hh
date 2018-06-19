#ifndef _NNFC_H264_IMAGE_H
#define _NNFC_H264_IMAGE_H

#include <cstdint>
#include <functional>
#include <vector>

#include "codec/mpeg.hh"
#include "codec/swizzle.hh"
#include "nnfc_CXXAPI.hh"
#include "tensor.hh"

namespace nnfc {

class H264ImageEncoder {
 private:
    codec::AVCEncoder encoder_;
    
 public:
  H264ImageEncoder(int quality);
  ~H264ImageEncoder() {}

  std::vector<uint8_t> forward(nn::Tensor<float, 3> input);
  nn::Tensor<float, 3> backward(nn::Tensor<float, 3> input);

  static nnfc::cxxapi::constructor_type_list initialization_params() {
    return {{"quantizer", typeid(int)}};
  }
};

class H264ImageDecoder {
 private:
    codec::AVCDecoder decoder_;

 public:
  H264ImageDecoder();
  ~H264ImageDecoder();

  nn::Tensor<float, 3> forward(std::vector<uint8_t> input);
  nn::Tensor<float, 3> backward(nn::Tensor<float, 3> input);

  static nnfc::cxxapi::constructor_type_list initialization_params() {
    return {};
  }
};
}

#endif  // _NNFC_H264_IMAGE_H
