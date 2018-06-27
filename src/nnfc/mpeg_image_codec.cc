#include "mpeg_image_codec.hh"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "tensor.hh"
using namespace std;

template <class Encoder>
nnfc::MPEGImageEncoder<Encoder>::MPEGImageEncoder(int quality)
    : encoder_(quality) {}

template <class Encoder>
vector<uint8_t> nnfc::MPEGImageEncoder<Encoder>::forward(
    nn::Tensor<float, 3> input) {
  const uint64_t dim0 = input.dimension(0);
  const uint64_t dim1 = input.dimension(1);
  const uint64_t dim2 = input.dimension(2);

  if (dim0 != 3) {
    throw std::runtime_error("The MPEGImageEncoder expects a 3-channel input");
  }

  const float min = input.minimum();
  const float max = input.maximum();

  vector<uint8_t> buffer(dim0 * dim1 * dim2);
  fill(buffer.begin(), buffer.end(), 0);

  // rescale the input to be from 0 to 255
  Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> input_q =
      ((input.tensor() - min) * (255 / (max - min))).cast<uint8_t>();

  std::memcpy(buffer.data(), &input_q(0, 0, 0), dim0 * dim1 * dim2);
  codec::RGBp_to_YUV420p converter(dim1, dim2);
  std::vector<uint8_t> yuv = converter.convert(buffer);

  // H264 compress
  vector<uint8_t> encoding = encoder_.encode(yuv, dim1, dim2, 3);
  // auto encoding = yuv;

  const uint8_t *min_bytes = reinterpret_cast<const uint8_t *>(&min);
  const uint8_t *max_bytes = reinterpret_cast<const uint8_t *>(&max);

  for (size_t i = 0; i < sizeof(float); i++) {
    encoding.push_back(min_bytes[i]);
  }
  for (size_t i = 0; i < sizeof(float); i++) {
    encoding.push_back(max_bytes[i]);
  }

  const uint8_t *dim0_bytes = reinterpret_cast<const uint8_t *>(&dim0);
  const uint8_t *dim1_bytes = reinterpret_cast<const uint8_t *>(&dim1);
  const uint8_t *dim2_bytes = reinterpret_cast<const uint8_t *>(&dim2);

  for (size_t i = 0; i < sizeof(uint64_t); i++) {
    encoding.push_back(dim0_bytes[i]);
  }
  for (size_t i = 0; i < sizeof(uint64_t); i++) {
    encoding.push_back(dim1_bytes[i]);
  }
  for (size_t i = 0; i < sizeof(uint64_t); i++) {
    encoding.push_back(dim2_bytes[i]);
  }

  return encoding;
}

template <class Encoder>
nn::Tensor<float, 3> nnfc::MPEGImageEncoder<Encoder>::backward(
    nn::Tensor<float, 3> input) {
  return input;
}

template <class Decoder>
nnfc::MPEGImageDecoder<Decoder>::MPEGImageDecoder() : decoder_() {}

template <class Decoder>
nn::Tensor<float, 3> nnfc::MPEGImageDecoder<Decoder>::forward(
    vector<uint8_t> input) {
  uint64_t dim0;
  uint64_t dim1;
  uint64_t dim2;
  uint8_t *dim0_bytes = reinterpret_cast<uint8_t *>(&dim0);
  uint8_t *dim1_bytes = reinterpret_cast<uint8_t *>(&dim1);
  uint8_t *dim2_bytes = reinterpret_cast<uint8_t *>(&dim2);

  size_t length = input.size();
  size_t dim0_offset = length - 3 * sizeof(uint64_t);
  size_t dim1_offset = length - 2 * sizeof(uint64_t);
  size_t dim2_offset = length - 1 * sizeof(uint64_t);
  for (size_t i = 0; i < sizeof(uint64_t); i++) {
    dim0_bytes[i] = input[i + dim0_offset];
    dim1_bytes[i] = input[i + dim1_offset];
    dim2_bytes[i] = input[i + dim2_offset];
  }

  float min;
  float max;
  uint8_t *min_bytes = reinterpret_cast<uint8_t *>(&min);
  uint8_t *max_bytes = reinterpret_cast<uint8_t *>(&max);
  size_t min_offset = length - 3 * sizeof(uint64_t) - 2 * sizeof(float);
  size_t max_offset = length - 3 * sizeof(uint64_t) - 1 * sizeof(float);
  for (size_t i = 0; i < sizeof(uint64_t); i++) {
    min_bytes[i] = input[i + min_offset];
    max_bytes[i] = input[i + max_offset];
  }

  vector<uint8_t> decoded = decoder_.decode(input, dim2, dim1);
  // vector<uint8_t> decoded;
  // for(size_t i = 0; i < input.size() - 3*sizeof(uint64_t) - 2*sizeof(float);
  // i++){
  //     decoded.push_back(input[i]);
  // }

  codec::YUV420p_to_RGBp deconverter(dim1, dim2);
  std::vector<uint8_t> rgb = deconverter.convert(decoded);

  nn::Tensor<uint8_t, 3> output_q(dim0, dim1, dim2);
  std::memcpy(&output_q(0, 0, 0), rgb.data(), dim0 * dim1 * dim2);

  nn::Tensor<float, 3> output(dim0, dim1, dim2);
  for (size_t row = 0; row < dim1; row++) {
    for (size_t col = 0; col < dim2; col++) {
      for (size_t channel = 0; channel < dim0; channel++) {
        output(channel, row, col) = static_cast<float>(
            ((output_q(channel, row, col) * (max - min)) / 255) + min);
      }
    }
  }

  return output;
}

template <class Decoder>
nn::Tensor<float, 3> nnfc::MPEGImageDecoder<Decoder>::backward(
    nn::Tensor<float, 3> input) {
  return input;
}

template class nnfc::MPEGImageEncoder<codec::AVCEncoder>;
template class nnfc::MPEGImageDecoder<codec::AVCDecoder>;
template class nnfc::MPEGImageEncoder<codec::HEIFEncoder>;
template class nnfc::MPEGImageDecoder<codec::HEIFDecoder>;
