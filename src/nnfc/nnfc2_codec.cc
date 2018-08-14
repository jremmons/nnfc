#include <any>
#include <cstdint>
#include <iostream>
#include <vector>

#include "nnfc2_codec.hh"
#include "tensor.hh"
#include "codec/utils.hh"

static constexpr int BLOCK_WIDTH = 8;

static constexpr int ZIGZAG_ORDER[][2] = {
    {0, 0},
    {1, 0}, {0, 1},
    {0, 2}, {1, 1}, {2, 0},
    {3, 0}, {2, 1}, {1, 2}, {0, 3},
    {0, 4}, {1, 3}, {2, 2}, {3, 1}, {4, 0},
    {5, 0}, {4, 1}, {3, 2}, {2, 3}, {1, 4}, {0, 5},
    {0, 6}, {1, 5}, {2, 4}, {3, 3}, {4, 2}, {5, 1}, {6, 0},
    {7, 0}, {6, 1}, {5, 2}, {4, 3}, {3, 4}, {2, 5}, {1, 6}, {0, 7},
    {1, 7}, {2, 6}, {3, 5}, {4, 4}, {5, 3}, {6, 2}, {7, 1}, 
    {7, 2}, {6, 3}, {5, 4}, {4, 5}, {3, 6}, {2, 7},
    {3, 7}, {4, 6}, {5, 5}, {6, 4}, {7, 3},  
    {7, 4}, {6, 5}, {5, 6}, {4, 7},
    {5, 7}, {6, 6}, {7, 5},
    {7, 6}, {6, 7},
    {7, 7}
};
static constexpr int ZIGZAG_LENGTH = sizeof(ZIGZAG_ORDER) / sizeof(ZIGZAG_ORDER[0]);


nnfc::NNFC2Encoder::NNFC2Encoder() :
    quantizer_(1) {}

nnfc::NNFC2Encoder::~NNFC2Encoder() {}

std::vector<uint8_t> nnfc::NNFC2Encoder::forward(const nn::Tensor<float, 3> t_input) const {

  const uint64_t dim0 = t_input.dimension(0);
  const uint64_t dim1 = t_input.dimension(1);
  const uint64_t dim2 = t_input.dimension(2);

  std::vector<uint8_t> encoding;

  // add header
  {
      uint64_t dim0_ = dim0;
      uint64_t dim1_ = dim1;
      uint64_t dim2_ = dim2;
      uint8_t *dim0_bytes = reinterpret_cast<uint8_t *>(&dim0_);
      uint8_t *dim1_bytes = reinterpret_cast<uint8_t *>(&dim1_);
      uint8_t *dim2_bytes = reinterpret_cast<uint8_t *>(&dim2_);
      for (size_t i = 0; i < sizeof(uint64_t); i++) {
        encoding.push_back(dim0_bytes[i]);
      }
      for (size_t i = 0; i < sizeof(uint64_t); i++) {
        encoding.push_back(dim1_bytes[i]);
      }
      for (size_t i = 0; i < sizeof(uint64_t); i++) {
        encoding.push_back(dim2_bytes[i]);
      }
  }
  
  // code block for doing 8-bit linear quantization. DCT for integers does not exist yet.
  // Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> q_input =
  //     ((t_input.tensor() - min) * (255 / (max - min))).cast<uint8_t>();
  // Eigen::Tensor<float, 3, Eigen::RowMajor> input_t =
  //     (q_input.cast<float>() * ((max - min) / 255)) + min;
  // nn::Tensor<float, 3> input(input_t);

  nn::Tensor<float, 3> input(std::move(codec::utils::dct(t_input, BLOCK_WIDTH)));

  for (size_t channel = 0; channel < dim0; channel++) {
      for (size_t row = 0; row < dim1 / BLOCK_WIDTH; row++) {
          for (size_t col = 0; col < dim2 / BLOCK_WIDTH; col++) {
              
              for (size_t i = 0; i < ZIGZAG_LENGTH; i++) {
                  const size_t row_offset = BLOCK_WIDTH*row + ZIGZAG_ORDER[i][0];
                  const size_t col_offset = BLOCK_WIDTH*col + ZIGZAG_ORDER[i][1];

                  // check if (value / quantizer) > 0, if not encode EOB and break
                  // otherwise encode the symbol as normal
                  input(channel, row_offset, col_offset); 
              }
          }
      }
  }
  
  
  // const float dct_min = dct_input.minimum();
  // const float dct_max = dct_input.maximum();
  // Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> q_input = ((255 * (dct_input.tensor() - dct_min)) / (quantizer_ * dct_max)).cast<uint8_t>();

  // arithmetic encode
  //  and serialize data
  
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

  size_t dim0_offset = 0 * sizeof(uint64_t);
  size_t dim1_offset = 1 * sizeof(uint64_t);
  size_t dim2_offset = 2 * sizeof(uint64_t);
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

        size_t offset = sizeof(float)*(dim1 * dim2 * i + dim2 * j + k) + 3*sizeof(uint64_t);
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
