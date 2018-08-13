#ifndef _NNFC_NNFC2_H
#define _NNFC_NNFC2_H

#include <cstdint>
#include <vector>

#include "nnfc_CXXAPI.hh"
#include "tensor.hh"

namespace nnfc {

class NNFC2Encoder {
 private:
  const int quantizer_nbins_;

 public:
  NNFC2Encoder();
  ~NNFC2Encoder();

  std::vector<uint8_t> forward(nn::Tensor<float, 3> input);
  nn::Tensor<float, 3> backward(nn::Tensor<float, 3> input);

  static nnfc::cxxapi::constructor_type_list initialization_params() {
    return {};
  }
};

class NNFC2Decoder {
 private:
 public:
  NNFC2Decoder();
  ~NNFC2Decoder();

  nn::Tensor<float, 3> forward(std::vector<uint8_t> input);
  nn::Tensor<float, 3> backward(nn::Tensor<float, 3> input);

  static nnfc::cxxapi::constructor_type_list initialization_params() {
    return {};
  }
};
}

#endif  // _NNFC_NNFC2_H
