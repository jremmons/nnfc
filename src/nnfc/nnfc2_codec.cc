#include <any>
#include <cstdint>
#include <iostream>
#include <vector>

#include "nnfc2_codec.hh"
#include "tensor.hh"
#include "codec/utils.hh"

static constexpr int BLOCK_WIDTH = 8;

nnfc::NNFC2Encoder::NNFC2Encoder() {}

nnfc::NNFC2Encoder::~NNFC2Encoder() {}

std::vector<uint8_t> nnfc::NNFC2Encoder::forward(const nn::Tensor<float, 3> t_input) const {

  uint64_t dim0 = t_input.dimension(0);
  uint64_t dim1 = t_input.dimension(1);
  uint64_t dim2 = t_input.dimension(2);


  // code block for doing 8-bit linear quantization. DCT for integers does not exist yet.
  // const float min = t_input.minimum();
  // const float max = t_input.maximum();
  // Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> q_input =
  //     ((t_input.tensor() - min) * (255 / (max - min))).cast<uint8_t>();
  // Eigen::Tensor<float, 3, Eigen::RowMajor> input_t =
  //     (q_input.cast<float>() * ((max - min) / 255)) + min;
  // nn::Tensor<float, 3> input(input_t);
                                                      
  nn::Tensor<float, 3> input(std::move(codec::utils::dct(t_input, BLOCK_WIDTH)));
  
  // quantize
  // arithmetic encode
  // add header and serialize data
  
  std::vector<uint8_t> encoding;

  for (size_t i = 0; i < dim0; i++) {
    for (size_t j = 0; j < dim1; j++) {
      for (size_t k = 0; k < dim2; k++) {
        float element = input(i, j, k);
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

nn::Tensor<float, 3> nnfc::NNFC2Encoder::backward(const nn::Tensor<float, 3> input) const {
   return input;
}

nnfc::NNFC2Decoder::NNFC2Decoder() {}

nnfc::NNFC2Decoder::~NNFC2Decoder() {}

nn::Tensor<float, 3> nnfc::NNFC2Decoder::forward(const std::vector<uint8_t> input) const {

  // deserialize
  // undo arithmetic encoding
  // unquantize
  // undo dct

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

  nn::Tensor<float, 3> f_output(dim0, dim1, dim2);

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

        f_output(i, j, k) = element;
      }
    }
  }

  nn::Tensor<float, 3> output(std::move(codec::utils::idct(f_output, BLOCK_WIDTH)));

  return output;
}

nn::Tensor<float, 3> nnfc::NNFC2Decoder::backward(const nn::Tensor<float, 3> input) const {
  return input;
}
