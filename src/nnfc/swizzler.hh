#ifndef _NNFC_SWIZZLER_H
#define _NNFC_SWIZZLER_H

#include <cstdint>
#include <vector>

#include "codec/swizzle.hh"
#include "nnfc_CXXAPI.hh"
#include "tensor.hh"

namespace nnfc {

class RGBSwizzlerEncoder {
 private:
  // codec::RGBp_to_YUV420p converter_;
  // codec::YUV420p_to_RGBp deconverter_;
  // codec::RGB24_to_YUV422p converter_;
  // codec::YUV422p_to_RGB24 deconverter_;

 public:
  RGBSwizzlerEncoder();
  ~RGBSwizzlerEncoder();

  std::vector<uint8_t> forward(nn::Tensor<float, 3> input);
  nn::Tensor<float, 3> backward(nn::Tensor<float, 3> input);

  static nnfc::cxxapi::constructor_type_list initialization_params() {
    return {};
  }
};

class RGBSwizzlerDecoder {
 private:
 public:
  RGBSwizzlerDecoder();
  ~RGBSwizzlerDecoder();

  nn::Tensor<float, 3> forward(std::vector<uint8_t> input);
  nn::Tensor<float, 3> backward(nn::Tensor<float, 3> input);

  static nnfc::cxxapi::constructor_type_list initialization_params() {
    return {};
  }
};
}

#endif  // _NNFC_SWIZZLER_H
