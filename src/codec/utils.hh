#ifndef _CODEC_UTILS_HH
#define _CODEC_UTILS_HH

#include "nn/tensor.hh"

namespace codec::utils {

nn::Tensor<float, 3> dct(const nn::Tensor<float, 3>& input, const int N = 8);
nn::Tensor<float, 3> idct(const nn::Tensor<float, 3>& input, const int N = 8);

// nn::Tensor<uint8_t, 3> dct(nn::Tensor<uint8_t, 3>& input, const int N = 8);
// nn::Tensor<uint8_t, 3> idct(nn::Tensor<uint8_t, 3>& input, const int N = 8);
}  // namespace codec::utils

#endif /* _CODEC_UTILS_HH */
