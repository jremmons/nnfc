#ifndef _NNFC_NNFC1_H
#define _NNFC_NNFC1_H

#include <turbojpeg.h>

#include <cstdint>
#include <vector>

#include "nn/tensor.hh"
#include "nnfc_CXXAPI.hh"

namespace nnfc {

class NNFC1Encoder {
 private:
  const int quantizer_nbins_;

 public:
  NNFC1Encoder();
  ~NNFC1Encoder();

  std::vector<uint8_t> forward(nn::Tensor<float, 3> input);
  nn::Tensor<float, 3> backward(nn::Tensor<float, 3> input);

  static nnfc::cxxapi::constructor_type_list initialization_params() {
    return {};
  }
};

class NNFC1Decoder {
 private:
 public:
  NNFC1Decoder();
  ~NNFC1Decoder();

  nn::Tensor<float, 3> forward(std::vector<uint8_t> input);
  nn::Tensor<float, 3> backward(nn::Tensor<float, 3> input);

  static nnfc::cxxapi::constructor_type_list initialization_params() {
    return {};
  }
};
}

#endif  // _NNFC_NNFC1_H
