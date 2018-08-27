#include "mpeg_codec.hh"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "nn/tensor.hh"

using namespace std;
using namespace nnfc;

template <class Encoder>
vector<uint8_t> MPEGEncoder<Encoder>::forward(nn::Tensor<float, 3> input) {
  const uint64_t dim0 = input.dimension(0);
  const uint64_t dim1 = input.dimension(1);
  const uint64_t dim2 = input.dimension(2);

  const float min = input.minimum();
  const float max = input.maximum();

  // create a square grid for the activations to go into
  const size_t image_chunks = ceil(sqrt(dim0));
  const size_t image_height = image_chunks * dim1;
  const size_t image_width = image_chunks * dim2;

  vector<uint8_t> buffer(image_height * image_width);
  fill(buffer.begin(), buffer.end(), 0);

  // compute the strides for laying out the data in memory
  const size_t row_channel_stride = dim1 * dim2;
  const size_t row_stride = image_chunks * dim2;
  const size_t channel_stride = dim2;

  Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> input_q =
      ((input.tensor() - min) * (255 / (max - min))).cast<uint8_t>();

  // swizzle the data into the right memory layout
  for (size_t row_channel = 0; row_channel < image_chunks * image_chunks;
       row_channel += image_chunks) {
    for (size_t row = 0; row < dim1; row++) {
      for (size_t channel = 0; channel < image_chunks; channel++) {
        if (row_channel + channel < dim0) {
          const size_t offset = row_channel_stride * row_channel +
                                row_stride * row + channel_stride * channel;
          memcpy(&buffer[offset], &input_q(row_channel + channel, row, 0),
                 dim2);
        }
      }
    }
  }

  // AVC compress
  vector<uint8_t> encoding =
      encoder_.encode(buffer, image_width, image_height, 1);

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
  const uint8_t *width_bytes = reinterpret_cast<const uint8_t *>(&image_width);
  const uint8_t *height_bytes = reinterpret_cast<const uint8_t *>(&image_width);

  for (size_t i = 0; i < sizeof(uint64_t); i++) {
    encoding.push_back(dim0_bytes[i]);
  }
  for (size_t i = 0; i < sizeof(uint64_t); i++) {
    encoding.push_back(dim1_bytes[i]);
  }
  for (size_t i = 0; i < sizeof(uint64_t); i++) {
    encoding.push_back(dim2_bytes[i]);
  }
  for (size_t i = 0; i < sizeof(uint64_t); i++) {
    encoding.push_back(width_bytes[i]);
  }
  for (size_t i = 0; i < sizeof(uint64_t); i++) {
    encoding.push_back(height_bytes[i]);
  }

  return encoding;
}

template <class Encoder>
nn::Tensor<float, 3> MPEGEncoder<Encoder>::backward(
    nn::Tensor<float, 3> input) {
  return input;
}

template <class Decoder>
nn::Tensor<float, 3> MPEGDecoder<Decoder>::forward(vector<uint8_t> input) {
  uint64_t dim0;
  uint64_t dim1;
  uint64_t dim2;
  uint64_t width;
  uint64_t height;

  uint8_t *dim0_bytes = reinterpret_cast<uint8_t *>(&dim0);
  uint8_t *dim1_bytes = reinterpret_cast<uint8_t *>(&dim1);
  uint8_t *dim2_bytes = reinterpret_cast<uint8_t *>(&dim2);
  uint8_t *width_bytes = reinterpret_cast<uint8_t *>(&width);
  uint8_t *height_bytes = reinterpret_cast<uint8_t *>(&height);

  size_t length = input.size();
  size_t dim0_offset = length - 5 * sizeof(uint64_t);
  size_t dim1_offset = length - 4 * sizeof(uint64_t);
  size_t dim2_offset = length - 3 * sizeof(uint64_t);
  size_t width_offset = length - 2 * sizeof(uint64_t);
  size_t height_offset = length - 1 * sizeof(uint64_t);
  for (size_t i = 0; i < sizeof(uint64_t); i++) {
    dim0_bytes[i] = input[i + dim0_offset];
    dim1_bytes[i] = input[i + dim1_offset];
    dim2_bytes[i] = input[i + dim2_offset];
    width_bytes[i] = input[i + width_offset];
    height_bytes[i] = input[i + height_offset];
  }

  float min;
  float max;
  uint8_t *min_bytes = reinterpret_cast<uint8_t *>(&min);
  uint8_t *max_bytes = reinterpret_cast<uint8_t *>(&max);
  size_t min_offset = length - 5 * sizeof(uint64_t) - 2 * sizeof(float);
  size_t max_offset = length - 5 * sizeof(uint64_t) - 1 * sizeof(float);
  for (size_t i = 0; i < sizeof(uint64_t); i++) {
    min_bytes[i] = input[i + min_offset];
    max_bytes[i] = input[i + max_offset];
  }

  input.resize(input.size() - 5 * sizeof(uint64_t) - 2 * sizeof(float));

  const size_t image_chunks = ceil(sqrt(dim0));
  vector<uint8_t> buffer = decoder_.decode(input, width, height);

  /* only Y channel is necessary */
  buffer.resize(width * height);

  // compute the strides for laying out the data in memory
  const size_t row_channel_stride = dim1 * dim2;
  const size_t row_stride = image_chunks * dim2;
  const size_t channel_stride = dim2;
  const size_t col_stride = 1;

  // swizzle the data into the right memory layout
  nn::Tensor<float, 3> output(dim0, dim1, dim2);

  for (size_t row_channel = 0; row_channel < image_chunks * image_chunks;
       row_channel += image_chunks) {
    for (size_t row = 0; row < dim1; row++) {
      for (size_t channel = 0; channel < image_chunks; channel++) {
        if (row_channel + channel < dim0) {
          for (size_t col = 0; col < dim2; col++) {
            const size_t offset = row_channel_stride * row_channel +
                                  row_stride * row + channel_stride * channel +
                                  col_stride * col;
            const double val = buffer[offset];
            output(row_channel + channel, row, col) =
                static_cast<float>((max - min) * (val / 255) + min);
          }
        }
      }
    }
  }

  return output;
}

template <class Decoder>
nn::Tensor<float, 3> MPEGDecoder<Decoder>::backward(
    nn::Tensor<float, 3> input) {
  return input;
}

template class MPEGEncoder<codec::AVCEncoder>;
template class MPEGDecoder<codec::AVCDecoder>;
template class MPEGEncoder<codec::HEIFEncoder>;
template class MPEGDecoder<codec::HEIFDecoder>;
