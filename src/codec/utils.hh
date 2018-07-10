#ifndef _CODEC_UTILS_HH
#define _CODEC_UTILS_HH

#include "nn/tensor.hh"

namespace codec::utils {

nn::Tensor<float, 3> dct(nn::Tensor<float, 3>& input, const int N = 8);
nn::Tensor<float, 3> idct(nn::Tensor<float, 3>& input, const int N = 8);
}

#endif /* _CODEC_UTILS_HH */
