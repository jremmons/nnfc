#ifndef _NNFC_NOOP_H
#define _NNFC_NOOP_H

#include <cstdint>
#include <vector>

#include "nn/tensor.hh"
#include "nnfc_CXXAPI.hh"

namespace nnfc {

class NoopEncoder {
 private:
 public:
  NoopEncoder();
  ~NoopEncoder();

  std::vector<uint8_t> forward(nn::Tensor<float, 3> input);
  nn::Tensor<float, 3> backward(nn::Tensor<float, 3> input);

  static nnfc::cxxapi::constructor_type_list initialization_params() {
    return {};
  }
};

class NoopDecoder {
 private:
 public:
  NoopDecoder();
  ~NoopDecoder();

  nn::Tensor<float, 3> forward(std::vector<uint8_t> input);
  nn::Tensor<float, 3> backward(nn::Tensor<float, 3> input);

  static nnfc::cxxapi::constructor_type_list initialization_params() {
    return {};
  }
};
}  // namespace nnfc

#endif  // _NNFC_NOOP_H
