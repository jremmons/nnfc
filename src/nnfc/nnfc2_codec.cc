#include <any>
#include <cstdint>
#include <iostream>
#include <vector>

#include "codec/utils.hh"
#include "nnfc2_codec.hh"
#include "tensor.hh"

static constexpr int BLOCK_WIDTH = 8;

// zigzag traversal order
// 0   1   5   6  14  15  27  28
// 
// 2   4   7  13  16  26  29  42
//
// 3   8  12  17  25  30  41  43
//
// 9  11  18  24  31  40  44  53
//
// 10  19  23  32  39  45  52  54
//
// 20  22  33  38  46  51  55  60
//
// 21  34  37  47  50  56  59  61
//
// 35  36  48  49  57  58  62  63
static constexpr int ZIGZAG_ORDER[][2] = {
    {0, 0},                                                          //
    {0, 1}, {1, 0},                                                  //
    {2, 0}, {1, 1}, {0, 2},                                          //
    {0, 3}, {1, 2}, {2, 1}, {3, 0},                                  //
    {4, 0}, {3, 1}, {2, 2}, {1, 3}, {0, 4},                          //
    {0, 5}, {1, 4}, {2, 3}, {3, 2}, {4, 1}, {5, 0},                  //
    {6, 0}, {5, 1}, {4, 2}, {3, 3}, {2, 4}, {1, 5}, {0, 6},          //
    {0, 7}, {1, 6}, {2, 5}, {3, 4}, {4, 3}, {5, 2}, {6, 1}, {7, 0},  //
    {7, 1}, {6, 2}, {5, 3}, {4, 4}, {3, 5}, {2, 6}, {1, 7},          //
    {2, 7}, {3, 6}, {4, 5}, {5, 4}, {6, 3}, {7, 2},                  //
    {7, 3}, {6, 4}, {5, 5}, {4, 6}, {3, 7},                          //
    {4, 7}, {5, 6}, {6, 5}, {7, 4},                                  //
    {7, 5}, {6, 6}, {5, 7},                                          //
    {6, 7}, {7, 6},                                                  //
    {7, 7}                                                           //
};
static constexpr int ZIGZAG_LENGTH =
    sizeof(ZIGZAG_ORDER) / sizeof(ZIGZAG_ORDER[0]);

// check if zigzag hits all elements
// static void check_zigzag() {
//   int check[8][8] = {{0}};
//   for (int i = 0; i < ZIGZAG_LENGTH; i++) {
//       int x = ZIGZAG_ORDER[i][0];
//       int y = ZIGZAG_ORDER[i][1];

//       check[x][y] += 1;
//   }
//   for (int i = 0; i < 8; i++) {
//       for (int j = 0; j < 8; j++) {
//           std::cout << check[i][j] << std::endl;
//           if (check[i][j] != 1) {
//               std::cout << "" << i << " " << j << " " << check[i][j] << std::endl;
//               throw std::runtime_error("check failed");
//           }
//       }
//   }
//   std::cout << "check passed" << std::endl;
// }

// Goes in order of zigzag pattern.
// Quantization value taken from https://github.com/libjpeg-turbo/ijg/blob/3040e5eaef76f270f58fba7ed87472d5c12c539f/jcparam.c#L68.
// Gist: (https://gist.github.com/jremmons/245506018f5933bf344c2e37ec40a24e#file-jpeg_quantization-c-L1)
// We use the luminance quantization values.
static constexpr int JPEG_QUANTIZATION[] = {
    16,                                  //
    11,  12,                             //
    14,  12,  10,                        //
    16,  14,  13,  14,                   //
    18,  17,  16,  19,  24,              //
    40,  26,  24,  22,  22, 24,          //
    49,  35,  37,  29,  40, 58, 51,      //
    61,  60,  57,  51,  56, 55, 64, 72,  //
    92,  78,  64,  68,  87, 69, 55,      //
    56,  80,  109, 81,  87, 95,          //
    98,  103, 104, 103, 62,              //
    77,  113, 121, 112,                  //
    100, 120, 92,                        //
    101, 103,                            //
    99                                   //
};
static constexpr int JPEG_QUANTIZATION_LENGTH =
    sizeof(JPEG_QUANTIZATION) / sizeof(JPEG_QUANTIZATION[0]);
static_assert(ZIGZAG_LENGTH == JPEG_QUANTIZATION_LENGTH);


nnfc::NNFC2Encoder::NNFC2Encoder() : quantizer_(1) {}

nnfc::NNFC2Encoder::~NNFC2Encoder() {}

std::vector<uint8_t> nnfc::NNFC2Encoder::forward(
  const nn::Tensor<float, 3> t_input) const {
  const uint64_t dim0 = t_input.dimension(0);
  const uint64_t dim1 = t_input.dimension(1);
  const uint64_t dim2 = t_input.dimension(2);

  assert(dim1 % BLOCK_WIDTH == 0);
  assert(dim2 % BLOCK_WIDTH == 0);
  
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

  // perform quantization first,
  // then DCT
  // then further quantize using DCT table
  // then encode in zigzag order
  // then arithmetic encode

  const float min = t_input.minimum();
  const float max = t_input.maximum();
  const float range = max - min;

  // quantize to 8-bits
  Eigen::Tensor<float, 3, Eigen::RowMajor> q1_input = (255 * (t_input.tensor() - min))  / (quantizer_ * range);
  nn::Tensor<float, 3> q_input(q1_input); 

  // do the dct
  // nn::Tensor<float, 3> dct_input(q_input);
  nn::Tensor<float, 3> dct_input(std::move(codec::utils::dct(q_input, BLOCK_WIDTH)));
  
  // std::cout << dct_input.maximum() << " " << dct_input.minimum() << std::endl;
  
  // discretize to int32
  Eigen::Tensor<int32_t, 3, Eigen::RowMajor> q2_input = dct_input.tensor().cast<int32_t>();
  nn::Tensor<int32_t, 3> input(q2_input);
  // Eigen::Tensor<float, 3, Eigen::RowMajor> q2_input = dct_input.tensor().cast<float>();
  // nn::Tensor<float, 3> input(q2_input);

  // add min and max to header
  {
    float min_ = min;
    float max_ = max;
    uint8_t *min_bytes = reinterpret_cast<uint8_t *>(&min_);
    uint8_t *max_bytes = reinterpret_cast<uint8_t *>(&max_);
    for (size_t i = 0; i < sizeof(float); i++) {
      encoding.push_back(min_bytes[i]);
    }
    for (size_t i = 0; i < sizeof(float); i++) {
      encoding.push_back(max_bytes[i]);
    }
  }

  // add quantizer to header
  {
    int32_t quantizer = quantizer_;
    uint8_t *quantizer_bytes = reinterpret_cast<uint8_t *>(&quantizer);
    for (size_t i = 0; i < sizeof(int32_t); i++) {
      encoding.push_back(quantizer_bytes[i]);
    }
  }

  // arithmetic encode and serialize data
  for (size_t channel = 0; channel < dim0; channel++) {
    for (size_t block_row = 0; block_row < dim1 / BLOCK_WIDTH; block_row++) {
      for (size_t block_col = 0; block_col < dim2 / BLOCK_WIDTH; block_col++) {
        for (size_t i = 0; i < ZIGZAG_LENGTH; i++) {
          const size_t row_offset =
              BLOCK_WIDTH * block_row + ZIGZAG_ORDER[i][0];
          const size_t col_offset =
              BLOCK_WIDTH * block_col + ZIGZAG_ORDER[i][1];

          int32_t element = input(channel, row_offset, col_offset) / (JPEG_QUANTIZATION[i]/2);
          //float element = input(channel, row_offset, col_offset) /* / (JPEG_QUANTIZATION[i]/4) */;
          
          // apply JPEG quantizer
          // check if (value / quantizer) > 0, if not encode 0 (or EOB and break)
          // otherwise encode the symbol as normal

          const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&element);
          
          encoding.push_back(bytes[0]);
          encoding.push_back(bytes[1]);
          encoding.push_back(bytes[2]);
          encoding.push_back(bytes[3]);
        }
      }
    }
  }

  return encoding;
}

nn::Tensor<float, 3> nnfc::NNFC2Encoder::backward(
    const nn::Tensor<float, 3> input) const {
  return input;
}

nnfc::NNFC2Decoder::NNFC2Decoder() {}

nnfc::NNFC2Decoder::~NNFC2Decoder() {}

nn::Tensor<float, 3> nnfc::NNFC2Decoder::forward(
    const std::vector<uint8_t> input) const {
  // read dims from header
  uint64_t dim0;
  uint64_t dim1;
  uint64_t dim2;
  {
    uint8_t *dim0_bytes = reinterpret_cast<uint8_t *>(&dim0);
    uint8_t *dim1_bytes = reinterpret_cast<uint8_t *>(&dim1);
    uint8_t *dim2_bytes = reinterpret_cast<uint8_t *>(&dim2);

    constexpr size_t dim0_offset = 0 * sizeof(uint64_t);
    constexpr size_t dim1_offset = 1 * sizeof(uint64_t);
    constexpr size_t dim2_offset = 2 * sizeof(uint64_t);
    for (size_t i = 0; i < sizeof(uint64_t); i++) {
      dim0_bytes[i] = input[i + dim0_offset];
      dim1_bytes[i] = input[i + dim1_offset];
      dim2_bytes[i] = input[i + dim2_offset];
    }
  }

  // read min and max from header
  float min;
  float max;
  {
    uint8_t *min_bytes = reinterpret_cast<uint8_t *>(&min);
    uint8_t *max_bytes = reinterpret_cast<uint8_t *>(&max);

    constexpr size_t min_offset = 3 * sizeof(uint64_t);
    constexpr size_t max_offset = sizeof(float) + 3 * sizeof(uint64_t);
    for (size_t i = 0; i < sizeof(float); i++) {
      min_bytes[i] = input[i + min_offset];
    }
    for (size_t i = 0; i < sizeof(float); i++) {
      max_bytes[i] = input[i + max_offset];
    }
  }
  const float range = max - min;

  // read quantizer from header
  int32_t quantizer;
  {
    uint8_t *quantizer_bytes = reinterpret_cast<uint8_t *>(&quantizer);

    constexpr size_t quantizer_offset =
        2 * sizeof(float) + 3 * sizeof(uint64_t);
    for (size_t i = 0; i < sizeof(int32_t); i++) {
      quantizer_bytes[i] = input[i + quantizer_offset];
    }
  }

  assert(dim1 % BLOCK_WIDTH == 0);
  assert(dim2 % BLOCK_WIDTH == 0);

  constexpr size_t header_offset =
      sizeof(int32_t) + 2 * sizeof(float) + 3 * sizeof(uint64_t);

  nn::Tensor<int32_t, 3> f_output(dim0, dim1, dim2);
  //nn::Tensor<float, 3> f_output(dim0, dim1, dim2);

  // undo arithmetic encoding and deserialize
  for (size_t channel = 0; channel < dim0; channel++) {
    for (size_t block_row = 0; block_row < dim1 / BLOCK_WIDTH; block_row++) {
      for (size_t block_col = 0; block_col < dim2 / BLOCK_WIDTH; block_col++) {
        for (size_t i = 0; i < ZIGZAG_LENGTH; i++) {
          const size_t row_offset =
              BLOCK_WIDTH * block_row + ZIGZAG_ORDER[i][0];
          const size_t col_offset =
              BLOCK_WIDTH * block_col + ZIGZAG_ORDER[i][1];

          const size_t offset =
              sizeof(int32_t) *
              (dim1 * dim2 * channel + BLOCK_WIDTH * dim2 * block_row +
               BLOCK_WIDTH * BLOCK_WIDTH * block_col + i) + header_offset;

          int32_t element;
          // float element;
          uint8_t* bytes = reinterpret_cast<uint8_t*>(&element);
          
          bytes[0] = input[offset];
          bytes[1] = input[offset + 1];
          bytes[2] = input[offset + 2];
          bytes[3] = input[offset + 3];
          
          f_output(channel, row_offset, col_offset) = (JPEG_QUANTIZATION[i]/2) * element;
          
        }
      }
    }
  }

  Eigen::Tensor<float, 3, Eigen::RowMajor> fp_output = f_output.tensor().cast<float>();
  
  // undo the dct
  // nn::Tensor<float, 3> idct_output(fp_output);
  nn::Tensor<float, 3> idct_output(
      std::move(codec::utils::idct(fp_output, BLOCK_WIDTH)));

  // dequantize from 8-bits
  Eigen::Tensor<float, 3, Eigen::RowMajor> dq1_output =
      (quantizer * range) * idct_output.tensor();
  Eigen::Tensor<float, 3, Eigen::RowMajor> dq2_output =
      ((1 / 255.f) * dq1_output) + min;

  // convert into nn::Tensor
  nn::Tensor<float, 3> output(dq2_output);

  return output;
}

nn::Tensor<float, 3> nnfc::NNFC2Decoder::backward(
    const nn::Tensor<float, 3> input) const {
  return input;
}
