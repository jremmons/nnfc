#ifndef _MPEG_IMAGE_CODEC_HH
#define _MPEG_IMAGE_CODEC_HH

#include <cstdint>
#include <functional>
#include <vector>

#include "codec/mpeg.hh"
#include "codec/swizzle.hh"
#include "nnfc_CXXAPI.hh"
#include "tensor.hh"

namespace nnfc {

template<class Encoder>
class MPEGImageEncoder {
 private:
    Encoder encoder_;

 public:
  MPEGImageEncoder(int quality);
  ~MPEGImageEncoder() {}

  std::vector<uint8_t> forward(nn::Tensor<float, 3> input);
  nn::Tensor<float, 3> backward(nn::Tensor<float, 3> input);

  static nnfc::cxxapi::constructor_type_list initialization_params() {
    return {{"quantizer", typeid(int)}};
  }
};

template<class Decoder>
class MPEGImageDecoder {
 private:
    Decoder decoder_;

 public:
  MPEGImageDecoder();
  ~MPEGImageDecoder() {}

  nn::Tensor<float, 3> forward(std::vector<uint8_t> input);
  nn::Tensor<float, 3> backward(nn::Tensor<float, 3> input);

  static nnfc::cxxapi::constructor_type_list initialization_params() {
    return {};
  }
};

using H264ImageEncoder = MPEGImageEncoder<codec::AVCEncoder>;
using H264ImageDecoder = MPEGImageDecoder<codec::AVCDecoder>;
using H265ImageEncoder = MPEGImageEncoder<codec::HEIFEncoder>;
using H265ImageDecoder = MPEGImageDecoder<codec::HEIFDecoder>;
}

#endif /* _MPEG_IMAGE_CODEC_HH */
