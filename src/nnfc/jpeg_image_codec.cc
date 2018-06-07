#include "jpeg_image_codec.hh"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "tensor.hh"
using namespace std;

nnfc::JPEGImageEncoder::JPEGImageEncoder(int quality) : encoder_(quality) {}

vector<uint8_t> nnfc::JPEGImageEncoder::forward(nn::Tensor<float, 3> input) {
  const uint64_t dim0 = input.dimension(0);
  const uint64_t dim1 = input.dimension(1);
  const uint64_t dim2 = input.dimension(2);

  if (dim0 != 3) {
    throw std::runtime_error("The JPEGImageEncoder expects a 3-channel input");
  }

  const float min = input.minimum();
  const float max = input.maximum();

  // create a buffer to put the swizzled pixels
  vector<uint8_t> buffer(dim0 * dim1 * dim2);
  fill(buffer.begin(), buffer.end(), 0);

  // rescale the input to be from 0 to 255
  Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> input_q =
      ((input.tensor() - min) * (255 / (max - min))).cast<uint8_t>();

  // swizzle the data into the right memory layout
  for (size_t row = 0; row < dim1; row++) {
    for (size_t col = 0; col < dim2; col++) {
      for (size_t channel = 0; channel < dim0; channel++) {
        const size_t offset = dim2 * dim0 * row + dim0 * col + channel;
        buffer[offset] = input_q(channel, row, col);
      }
    }
  }

  // JPEG compress
  vector<uint8_t> encoding = encoder_.encode(buffer, dim2, dim1, 3);

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

nn::Tensor<float, 3> nnfc::JPEGImageEncoder::backward(
    nn::Tensor<float, 3> input) {
  return input;
}

nnfc::JPEGImageDecoder::JPEGImageDecoder()
    : jpeg_decompressor(tjInitDecompress(), [](void *ptr) { tjDestroy(ptr); }) {
}

nnfc::JPEGImageDecoder::~JPEGImageDecoder() {}

nn::Tensor<float, 3> nnfc::JPEGImageDecoder::forward(vector<uint8_t> input) {
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

  const long unsigned int jpeg_size =
      input.size() - 3 * sizeof(uint64_t) - 2 * sizeof(float);

  int jpegSubsamp, width, height;
  tjDecompressHeader2(jpeg_decompressor.get(), input.data(), jpeg_size, &width,
                      &height, &jpegSubsamp);

  vector<uint8_t> buffer(width * height * 3);

  tjDecompress2(jpeg_decompressor.get(), input.data(), jpeg_size, buffer.data(),
                width, 0 /*pitch*/, height, TJPF_RGB, TJFLAG_FASTDCT);

  // swizzle the data into the right memory layout
  nn::Tensor<float, 3> output(dim0, dim1, dim2);

  for (size_t row = 0; row < dim1; row++) {
    for (size_t col = 0; col < dim2; col++) {
      for (size_t channel = 0; channel < dim0; channel++) {
        const size_t offset = dim2 * dim0 * row + dim0 * col + channel;
        const float val = buffer[offset];
        output(channel, row, col) =
            static_cast<float>(((val * (max - min)) / 255) + min);
      }
    }
  }

  return output;
}

nn::Tensor<float, 3> nnfc::JPEGImageDecoder::backward(
    nn::Tensor<float, 3> input) {
  return input;
}
