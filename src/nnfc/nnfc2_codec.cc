#include <any>
#include <cstdint>
#include <iostream>
#include <vector>

#include <chrono>

#include "codec/arithmetic_coder.hh"
#include "codec/fastdct.hh"
#include "codec/utils.hh"
#include "nn/tensor.hh"

#include "nnfc2_codec.hh"

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
//               std::cout << "" << i << " " << j << " " << check[i][j] <<
//               std::endl;
//               throw std::runtime_error("check failed");
//           }
//       }
//   }
//   std::cout << "check passed" << std::endl;
// }

// Goes in order of zigzag pattern.
// Quantization value taken from
// https://github.com/libjpeg-turbo/ijg/blob/3040e5eaef76f270f58fba7ed87472d5c12c539f/jcparam.c#L68.
// Gist:
// (https://gist.github.com/jremmons/245506018f5933bf344c2e37ec40a24e#file-jpeg_quantization-c-L1)
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

// inclusive range of values the DCT coefficients can take on
static constexpr int32_t DCT_MIN = -64;
static constexpr int32_t DCT_MAX = 64;

nnfc::NNFC2Encoder::NNFC2Encoder() : quality_(48) {}

nnfc::NNFC2Encoder::~NNFC2Encoder() {}

std::vector<uint8_t> nnfc::NNFC2Encoder::forward(
    const nn::Tensor<float, 3> t_input) const {
  const uint64_t dim0 = t_input.dimension(0);
  const uint64_t dim1 = t_input.dimension(1);
  const uint64_t dim2 = t_input.dimension(2);

  assert(dim1 % BLOCK_WIDTH == 0);
  assert(dim2 % BLOCK_WIDTH == 0);

  const float min = t_input.minimum();
  const float max = t_input.maximum();
  const float range = max - min;

  // auto quantize_t1 = std::chrono::high_resolution_clock::now();

  // quantize to 8-bits
  Eigen::Tensor<float, 3, Eigen::RowMajor> q1_input =
      (255.f * (t_input.tensor() - min)) / range;
  // Eigen::Tensor<float, 3, Eigen::RowMajor> q1_input =
  //     (127.f * (t_input.tensor() - min)) / range;
  // Eigen::Tensor<float, 3, Eigen::RowMajor> q1_input = (63.f *
  // (t_input.tensor() - min)) / range - 32.f;
  nn::Tensor<float, 3> q_input(q1_input);

  nn::Tensor<int16_t, 3> dct_in(q_input.dimension(0), q_input.dimension(1),
                                q_input.dimension(2));

  // round to nearest
  for (size_t channel = 0; channel < dim0; channel++) {
    for (size_t row = 0; row < dim1; row++) {
      for (size_t col = 0; col < dim2; col++) {
        const float value = std::round(q_input(channel, row, col));
        dct_in(channel, row, col) = static_cast<int16_t>(value);
      }
    }
  }
  // auto quantize_t2 = std::chrono::high_resolution_clock::now();
  // std::cout << "quantize time: "
            << std::chrono::duration_cast<std::chrono::duration<double>>(
                   quantize_t2 - quantize_t1)
                   .count()
            << std::endl;

  // do the dct
  // nn::Tensor<int16_t, 3> dct_out(dct_in);
  codec::FastDCT dct;
  // auto dct_t1 = std::chrono::high_resolution_clock::now();
  nn::Tensor<int16_t, 3> dct_out = dct(dct_in);
  // auto dct_t2 = std::chrono::high_resolution_clock::now();
  // std::cout << "dct time: "
            << std::chrono::duration_cast<std::chrono::duration<double>>(
                   dct_t2 - dct_t1)
                   .count()
            << std::endl;

  assert(quality_ > 0);
  assert(quality_ <= 100);
  const float scale = quality_ < 50 ? 50.f / quality_ : (100.f - quality_) / 50;

  // codec::DummyArithmeticEncoder encoder;
  codec::ArithmeticEncoder<codec::SimpleAdaptiveModel> encoder(DCT_MAX -
                                                               DCT_MIN + 1);

  // arithmetic encode and serialize data
  // auto encode_t1 = std::chrono::high_resolution_clock::now();
  for (size_t channel = 0; channel < dim0; channel++) {
    for (size_t block_row = 0; block_row < dim1 / BLOCK_WIDTH; block_row++) {
      for (size_t block_col = 0; block_col < dim2 / BLOCK_WIDTH; block_col++) {
        for (size_t i = 0; i < ZIGZAG_LENGTH; i++) {
          const size_t row_offset =
              BLOCK_WIDTH * block_row + ZIGZAG_ORDER[i][0];
          const size_t col_offset =
              BLOCK_WIDTH * block_col + ZIGZAG_ORDER[i][1];

          float scalef = scale * JPEG_QUANTIZATION[i];
          if (scalef < 1) {
            scalef = 1;
          }
          const float valf =
              static_cast<float>(dct_out(channel, row_offset, col_offset));
          const int32_t element =
              static_cast<int32_t>(std::round(valf / scalef));

          assert(element >= DCT_MIN);
          assert(element < DCT_MAX);

          const int symbol = element - DCT_MIN;
          assert(symbol >= 0);
          assert(symbol < (DCT_MAX - DCT_MIN + 1));

          encoder.encode_symbol(static_cast<uint32_t>(symbol));
        }
      }
    }
  }
  // auto encode_t2 = std::chrono::high_resolution_clock::now();
  // std::cout << "encode time: "
            << std::chrono::duration_cast<std::chrono::duration<double>>(
                   encode_t2 - encode_t1)
                   .count()
            << std::endl;

  std::vector<char> encoding = encoder.finish();

  // auto serialize_t1 = std::chrono::high_resolution_clock::now();
  // add footer
  {
    uint64_t dim0_ = dim0;
    uint64_t dim1_ = dim1;
    uint64_t dim2_ = dim2;
    char *dim0_bytes = reinterpret_cast<char *>(&dim0_);
    char *dim1_bytes = reinterpret_cast<char *>(&dim1_);
    char *dim2_bytes = reinterpret_cast<char *>(&dim2_);
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

  // add min and max to footer
  {
    float min_ = min;
    float max_ = max;
    char *min_bytes = reinterpret_cast<char *>(&min_);
    char *max_bytes = reinterpret_cast<char *>(&max_);
    for (size_t i = 0; i < sizeof(float); i++) {
      encoding.push_back(min_bytes[i]);
    }
    for (size_t i = 0; i < sizeof(float); i++) {
      encoding.push_back(max_bytes[i]);
    }
  }

  // add quality to footer
  {
    int32_t quality = quality_;
    char *quality_bytes = reinterpret_cast<char *>(&quality);
    for (size_t i = 0; i < sizeof(int32_t); i++) {
      encoding.push_back(quality_bytes[i]);
    }
  }

  std::vector<uint8_t> encoding_(
      reinterpret_cast<uint8_t *>(encoding.data()),
      reinterpret_cast<uint8_t *>(encoding.data()) + encoding.size());

  // auto serialize_t2 = std::chrono::high_resolution_clock::now();
  // std::cout << "serialize time: "
            << std::chrono::duration_cast<std::chrono::duration<double>>(
                   serialize_t2 - serialize_t1)
                   .count()
            << std::endl;

  return encoding_;
}

nn::Tensor<float, 3> nnfc::NNFC2Encoder::backward(
    const nn::Tensor<float, 3> input) const {
  return input;
}

nnfc::NNFC2Decoder::NNFC2Decoder() {}

nnfc::NNFC2Decoder::~NNFC2Decoder() {}

nn::Tensor<float, 3> nnfc::NNFC2Decoder::forward(
    const std::vector<uint8_t> input) const {
  const size_t input_size = input.size();

  // read dims from footer
  uint64_t dim0;
  uint64_t dim1;
  uint64_t dim2;
  {
    uint8_t *dim0_bytes = reinterpret_cast<uint8_t *>(&dim0);
    uint8_t *dim1_bytes = reinterpret_cast<uint8_t *>(&dim1);
    uint8_t *dim2_bytes = reinterpret_cast<uint8_t *>(&dim2);

    const size_t dim0_offset = input_size - 3 * sizeof(uint64_t) -
                               2 * sizeof(float) - 1 * sizeof(int32_t);
    const size_t dim1_offset = input_size - 2 * sizeof(uint64_t) -
                               2 * sizeof(float) - 1 * sizeof(int32_t);
    const size_t dim2_offset = input_size - 1 * sizeof(uint64_t) -
                               2 * sizeof(float) - 1 * sizeof(int32_t);
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

    const size_t min_offset =
        input_size - 2 * sizeof(float) - 1 * sizeof(int32_t);
    const size_t max_offset =
        input_size - 1 * sizeof(float) - 1 * sizeof(int32_t);
    for (size_t i = 0; i < sizeof(float); i++) {
      min_bytes[i] = input[i + min_offset];
    }
    for (size_t i = 0; i < sizeof(float); i++) {
      max_bytes[i] = input[i + max_offset];
    }
  }
  const float range = max - min;

  // read quality from header
  int32_t quality;
  {
    uint8_t *quality_bytes = reinterpret_cast<uint8_t *>(&quality);

    const size_t quality_offset = input_size - 1 * sizeof(int32_t);

    for (size_t i = 0; i < sizeof(int32_t); i++) {
      quality_bytes[i] = input[i + quality_offset];
    }
  }

  assert(dim1 % BLOCK_WIDTH == 0);
  assert(dim2 % BLOCK_WIDTH == 0);

  nn::Tensor<int16_t, 3> fp_output(dim0, dim1, dim2);

  assert(quality > 0);
  assert(quality <= 100);
  const float scale = quality < 50 ? 50.f / quality : (100.f - quality) / 50;

  std::vector<char> encoding_(reinterpret_cast<const char *>(input.data()),
                              reinterpret_cast<const char *>(input.data()) +
                                  input_size - 3 * sizeof(uint64_t) -
                                  2 * sizeof(float) - 1 * sizeof(int32_t));

  codec::ArithmeticDecoder<codec::SimpleAdaptiveModel> decoder(
      encoding_, DCT_MAX - DCT_MIN + 1);
  // codec::DummyArithmeticDecoder decoder(encoding_);

  // double time = 0;
  // size_t count = 0;

  // undo arithmetic encoding and deserialize
  for (size_t channel = 0; channel < dim0; channel++) {
    for (size_t block_row = 0; block_row < dim1 / BLOCK_WIDTH; block_row++) {
      for (size_t block_col = 0; block_col < dim2 / BLOCK_WIDTH; block_col++) {
        for (size_t i = 0; i < ZIGZAG_LENGTH; i++) {
          const size_t row_offset =
              BLOCK_WIDTH * block_row + ZIGZAG_ORDER[i][0];
          const size_t col_offset =
              BLOCK_WIDTH * block_col + ZIGZAG_ORDER[i][1];

          // auto t1 = std::chrono::high_resolution_clock::now();
          uint32_t symbol = decoder.decode_symbol();
          // auto t2 = std::chrono::high_resolution_clock::now();
          // count += 1;
          // time +=
          //     std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
          //         .count();

          assert(symbol < (DCT_MAX - DCT_MIN + 1));
          int32_t element = symbol + DCT_MIN;

          float scalef = scale * JPEG_QUANTIZATION[i];
          if (scalef < 1) {
            scalef = 1;
          }

          const float value = scalef * element;

          // clip if necessary
          // if (value < DCT_MIN) {
          //     value = DCT_MIN;
          // }
          // if (value >= DCT_MAX) {
          //     value = DCT_MAX;
          // }
          // assert(value >= DCT_MIN);
          // assert(value <= DCT_MAX);

          fp_output(channel, row_offset, col_offset) = std::round(value);
        }
      }
    }
  }

  // std::cout << "total time [decode] (count=" << time  << "): " << time <<
  // std::endl; std::cout << "avg time [decode]: " << time / count << std::endl;

  // undo the dct
  // nn::Tensor<int16_t, 3> idct_output(fp_output);
  codec::FastIDCT idct;
  nn::Tensor<uint8_t, 3> idct_output = idct(fp_output);
  // std::cout << (int)idct_output.minimum() << std::endl;
  // std::cout << (int)idct_output.maximum() << std::endl;
  // nn::Tensor<float, 3> idct_output(
  //     std::move(codec::utils::idct(fp_output, BLOCK_WIDTH)));

  // dequantize from 8-bits
  Eigen::Tensor<float, 3, Eigen::RowMajor> dq1_output =
      range * idct_output.tensor().cast<float>();
  Eigen::Tensor<float, 3, Eigen::RowMajor> dq2_output =
      ((1 / 255.f) * dq1_output) + min;

  // Eigen::Tensor<float, 3, Eigen::RowMajor> dq1_output =
  //     range * (idct_output.tensor().cast<float>());
  // Eigen::Tensor<float, 3, Eigen::RowMajor> dq2_output =
  //     ((1 / 127.f) * dq1_output) + min;

  // Eigen::Tensor<float, 3, Eigen::RowMajor> dq1_output =
  //     range * (idct_output.tensor() + 32.f);
  // Eigen::Tensor<float, 3, Eigen::RowMajor> dq2_output =
  //     ((1 / 63.f) * dq1_output) + min;

  // convert into nn::Tensor
  nn::Tensor<float, 3> output(dq2_output);

  return output;
}

nn::Tensor<float, 3> nnfc::NNFC2Decoder::backward(
    const nn::Tensor<float, 3> input) const {
  return input;
}
