#ifndef _NNFC_NNFC2_H
#define _NNFC_NNFC2_H

#include <cstdint>
#include <vector>

#include "nn/tensor.hh"
#include "nnfc_CXXAPI.hh"

namespace nnfc {

class NNFC2Encoder {
 private:
  const int32_t quality_;

 public:
  NNFC2Encoder();
  ~NNFC2Encoder();

  std::vector<uint8_t> forward(const nn::Tensor<float, 3> input) const;
  nn::Tensor<float, 3> backward(const nn::Tensor<float, 3> input) const;

  static nnfc::cxxapi::constructor_type_list initialization_params() {
    return {};
  }
};

class NNFC2Decoder {
 private:
 public:
  NNFC2Decoder();
  ~NNFC2Decoder();

  nn::Tensor<float, 3> forward(const std::vector<uint8_t> input) const;
  nn::Tensor<float, 3> backward(const nn::Tensor<float, 3> input) const;

  static nnfc::cxxapi::constructor_type_list initialization_params() {
    return {};
  }
};
}

#endif  // _NNFC_NNFC2_H
