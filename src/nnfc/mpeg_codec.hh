#ifndef _NNFC_AVC_CODEC_HH
#define _NNFC_AVC_CODEC_HH

#include <cstdint>
#include <functional>
#include <vector>

extern "C" {
#include <jpeglib.h>
}

#include <turbojpeg.h>

#include "codec/mpeg.hh"
#include "nn/tensor.hh"
#include "nnfc_CXXAPI.hh"

namespace nnfc {

template <class Encoder>
class MPEGEncoder {
 private:
  Encoder encoder_;

 public:
  MPEGEncoder(int quantizer) : encoder_(quantizer) {}
  ~MPEGEncoder() {}

  std::vector<uint8_t> forward(nn::Tensor<float, 3> input);
  nn::Tensor<float, 3> backward(nn::Tensor<float, 3> input);

  static nnfc::cxxapi::constructor_type_list initialization_params() {
    return {{"quantizer", typeid(int)}};
  }
};

template <class Decoder>
class MPEGDecoder {
 private:
  Decoder decoder_{};

 public:
  MPEGDecoder() {}
  ~MPEGDecoder() {}

  nn::Tensor<float, 3> forward(std::vector<uint8_t> input);
  nn::Tensor<float, 3> backward(nn::Tensor<float, 3> input);

  static nnfc::cxxapi::constructor_type_list initialization_params() {
    return {};
  }
};

using AVCEncoder = MPEGEncoder<codec::AVCEncoder>;
using AVCDecoder = MPEGDecoder<codec::AVCDecoder>;
using HEIFEncoder = MPEGEncoder<codec::HEIFEncoder>;
using HEIFDecoder = MPEGDecoder<codec::HEIFDecoder>;
}  // namespace nnfc

#endif  // _NNFC_AVC_CODEC_HH
