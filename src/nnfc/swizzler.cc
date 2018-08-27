#include <any>
#include <cstdint>
#include <iostream>
#include <vector>

#include "nn/tensor.hh"
#include "swizzler.hh"

nnfc::RGBSwizzlerEncoder::RGBSwizzlerEncoder() {}

nnfc::RGBSwizzlerEncoder::~RGBSwizzlerEncoder() {}

std::vector<uint8_t> nnfc::RGBSwizzlerEncoder::forward(
    nn::Tensor<float, 3> input) {
  uint64_t dim0 = input.dimension(0);
  uint64_t dim1 = input.dimension(1);
  uint64_t dim2 = input.dimension(2);

  if (dim0 != 3) {
    throw std::runtime_error("Expects a 3-channel input");
  }

  const float min = input.minimum();
  const float max = input.maximum();

  // create a buffer to put the swizzled pixels
  std::vector<uint8_t> buffer(dim0 * dim1 * dim2);
  fill(buffer.begin(), buffer.end(), 0);

  // rescale the input to be from 0 to 255
  Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> input_q =
      ((input.tensor() - min) * (255 / (max - min))).cast<uint8_t>();

  std::memcpy(buffer.data(), &input_q(0, 0, 0), dim0 * dim1 * dim2);

  codec::RGBp_to_YUV420p converter(dim1, dim2);
  std::vector<uint8_t> yuv = converter.convert(buffer);

  codec::YUV420p_to_RGBp deconverter(dim1, dim2);
  std::vector<uint8_t> rgb = deconverter.convert(yuv);

  std::memcpy(&input_q(0, 0, 0), rgb.data(), dim0 * dim1 * dim2);

  std::vector<uint8_t> encoding;
  for (size_t i = 0; i < dim0; i++) {
    for (size_t j = 0; j < dim1; j++) {
      for (size_t k = 0; k < dim2; k++) {
        float element = input_q(i, j, k);
        uint8_t *bytes = reinterpret_cast<uint8_t *>(&element);

        encoding.push_back(bytes[0]);
        encoding.push_back(bytes[1]);
        encoding.push_back(bytes[2]);
        encoding.push_back(bytes[3]);
      }
    }
  }

  uint8_t *dim0_bytes = reinterpret_cast<uint8_t *>(&dim0);
  uint8_t *dim1_bytes = reinterpret_cast<uint8_t *>(&dim1);
  uint8_t *dim2_bytes = reinterpret_cast<uint8_t *>(&dim2);
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

nn::Tensor<float, 3> nnfc::RGBSwizzlerEncoder::backward(
    nn::Tensor<float, 3> input) {
  return input;
}

nnfc::RGBSwizzlerDecoder::RGBSwizzlerDecoder() {}

nnfc::RGBSwizzlerDecoder::~RGBSwizzlerDecoder() {}

nn::Tensor<float, 3> nnfc::RGBSwizzlerDecoder::forward(
    std::vector<uint8_t> input) {
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

  nn::Tensor<float, 3> output(dim0, dim1, dim2);

  for (size_t i = 0; i < dim0; i++) {
    for (size_t j = 0; j < dim1; j++) {
      for (size_t k = 0; k < dim2; k++) {
        float element;
        uint8_t *bytes = reinterpret_cast<uint8_t *>(&element);

        size_t offset = sizeof(float) * (dim1 * dim2 * i + dim2 * j + k);
        bytes[0] = input[offset];
        bytes[1] = input[offset + 1];
        bytes[2] = input[offset + 2];
        bytes[3] = input[offset + 3];

        output(i, j, k) = element;
      }
    }
  }

  return output;
}

nn::Tensor<float, 3> nnfc::RGBSwizzlerDecoder::backward(
    nn::Tensor<float, 3> input) {
  return input;
}
