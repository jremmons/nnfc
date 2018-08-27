#include "jpeg_codec.hh"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "nn/tensor.hh"
using namespace std;

nnfc::JPEGEncoder::JPEGEncoder(int quality) : encoder_(quality) {}

vector<uint8_t> nnfc::JPEGEncoder::forward(nn::Tensor<float, 3> input) {
  const uint64_t dim0 = input.dimension(0);
  const uint64_t dim1 = input.dimension(1);
  const uint64_t dim2 = input.dimension(2);

  const float min = input.minimum();
  const float max = input.maximum();

  // create a square grid for the activations to go into
  const size_t jpeg_chunks = ceil(sqrt(dim0));
  const size_t jpeg_height = jpeg_chunks * dim1;
  const size_t jpeg_width = jpeg_chunks * dim2;

  vector<uint8_t> buffer(jpeg_height * jpeg_width);
  fill(buffer.begin(), buffer.end(), 0);

  // compute the strides for laying out the data in memory
  const size_t row_channel_stride = dim1 * dim2;
  const size_t row_stride = jpeg_chunks * dim2;
  const size_t channel_stride = dim2;

  Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> input_q =
      ((input.tensor() - min) * (255 / (max - min))).cast<uint8_t>();

  // swizzle the data into the right memory layout
  for (size_t row_channel = 0; row_channel < jpeg_chunks * jpeg_chunks;
       row_channel += jpeg_chunks) {
    for (size_t row = 0; row < dim1; row++) {
      for (size_t channel = 0; channel < jpeg_chunks; channel++) {
        if (row_channel + channel < dim0) {
          const size_t offset = row_channel_stride * row_channel +
                                row_stride * row + channel_stride * channel;
          memcpy(&buffer[offset], &input_q(row_channel + channel, row, 0),
                 dim2);
        }
      }
    }
  }

  // JPEG compress
  vector<uint8_t> encoding =
      encoder_.encode(buffer, jpeg_width, jpeg_height, 1);

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

nn::Tensor<float, 3> nnfc::JPEGEncoder::backward(nn::Tensor<float, 3> input) {
  return input;
}

nnfc::JPEGDecoder::JPEGDecoder()
    : jpeg_decompressor(tjInitDecompress(), [](void *ptr) { tjDestroy(ptr); }) {
}

nnfc::JPEGDecoder::~JPEGDecoder() {}

nn::Tensor<float, 3> nnfc::JPEGDecoder::forward(vector<uint8_t> input) {
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

  const size_t jpeg_chunks = ceil(sqrt(dim0));
  const long unsigned int jpeg_size =
      input.size() - 3 * sizeof(uint64_t) - 2 * sizeof(float);

  int jpegSubsamp, width, height;
  tjDecompressHeader2(jpeg_decompressor.get(), input.data(), jpeg_size, &width,
                      &height, &jpegSubsamp);

  vector<uint8_t> buffer(width * height);

  tjDecompress2(jpeg_decompressor.get(), input.data(), jpeg_size, buffer.data(),
                width, 0 /*pitch*/, height, TJPF_GRAY, TJFLAG_FASTDCT);

  // compute the strides for laying out the data in memory
  const size_t row_channel_stride = dim1 * dim2;
  const size_t row_stride = jpeg_chunks * dim2;
  const size_t channel_stride = dim2;
  const size_t col_stride = 1;

  // swizzle the data into the right memory layout
  nn::Tensor<float, 3> output(dim0, dim1, dim2);

  for (size_t row_channel = 0; row_channel < jpeg_chunks * jpeg_chunks;
       row_channel += jpeg_chunks) {
    for (size_t row = 0; row < dim1; row++) {
      for (size_t channel = 0; channel < jpeg_chunks; channel++) {
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

nn::Tensor<float, 3> nnfc::JPEGDecoder::backward(nn::Tensor<float, 3> input) {
  return input;
}
