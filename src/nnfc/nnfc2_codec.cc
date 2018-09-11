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
            // << std::chrono::duration_cast<std::chrono::duration<double>>(
            //        quantize_t2 - quantize_t1)
            //        .count()
            // << std::endl;

  // do the dct
  // nn::Tensor<int16_t, 3> dct_out(dct_in);
  codec::FastDCT dct;
  // auto dct_t1 = std::chrono::high_resolution_clock::now();
  nn::Tensor<int16_t, 3> dct_out = dct(dct_in);
  // auto dct_t2 = std::chrono::high_resolution_clock::now();
  // std::cout << "dct time: "
            // << std::chrono::duration_cast<std::chrono::duration<double>>(
            //        dct_t2 - dct_t1)
            //        .count()
            // << std::endl;

  assert(quality_ > 0);
  assert(quality_ <= 100);
  const float scale = quality_ < 50 ? 50.f / quality_ : (100.f - quality_) / 50;

  // codec::DummyArithmeticEncoder encoder;
  /*  codec::ArithmeticEncoder<codec::SimpleAdaptiveModel> encoder(     \
   "{\"denominator\":32899,\"num_symbols\":128,\"sym_0_lower\":0,\"sym_0_upper\":1,\"sym_100_lower\":32868,\"sym_100_upper\":32869,\"sym_101_lower\":32869,\"sym_101_upper\":32870,\"sym_102_lower\":32870,\"sym_102_upper\":32871,\"sym_103_lower\":32871,\"sym_103_upper\":32872,\"sym_104_lower\":32872,\"sym_104_upper\":32873,\"sym_105_lower\":32873,\"sym_105_upper\":32874,\"sym_106_lower\":32874,\"sym_106_upper\":32875,\"sym_107_lower\":32875,\"sym_107_upper\":32876,\"sym_108_lower\":32876,\"sym_108_upper\":32877,\"sym_109_lower\":32877,\"sym_109_upper\":32878,\"sym_10_lower\":10,\"sym_10_upper\":11,\"sym_110_lower\":32878,\"sym_110_upper\":32879,\"sym_111_lower\":32879,\"sym_111_upper\":32880,\"sym_112_lower\":32880,\"sym_112_upper\":32881,\"sym_113_lower\":32881,\"sym_113_upper\":32882,\"sym_114_lower\":32882,\"sym_114_upper\":32883,\"sym_115_lower\":32883,\"sym_115_upper\":32884,\"sym_116_lower\":32884,\"sym_116_upper\":32885,\"sym_117_lower\":32885,\"sym_117_upper\":32886,\"sym_118_lower\":32886,\"sym_118_upper\":32887,\"sym_119_lower\":32887,\"sym_119_upper\":32888,\"sym_11_lower\":11,\"sym_11_upper\":12,\"sym_120_lower\":32888,\"sym_120_upper\":32889,\"sym_121_lower\":32889,\"sym_121_upper\":32890,\"sym_122_lower\":32890,\"sym_122_upper\":32891,\"sym_123_lower\":32891,\"sym_123_upper\":32892,\"sym_124_lower\":32892,\"sym_124_upper\":32893,\"sym_125_lower\":32893,\"sym_125_upper\":32894,\"sym_126_lower\":32894,\"sym_126_upper\":32895,\"sym_127_lower\":32895,\"sym_127_upper\":32896,\"sym_12_lower\":12,\"sym_12_upper\":13,\"sym_13_lower\":13,\"sym_13_upper\":14,\"sym_14_lower\":14,\"sym_14_upper\":15,\"sym_15_lower\":15,\"sym_15_upper\":16,\"sym_16_lower\":16,\"sym_16_upper\":17,\"sym_17_lower\":17,\"sym_17_upper\":18,\"sym_18_lower\":18,\"sym_18_upper\":19,\"sym_19_lower\":19,\"sym_19_upper\":20,\"sym_1_lower\":1,\"sym_1_upper\":2,\"sym_20_lower\":20,\"sym_20_upper\":21,\"sym_21_lower\":21,\"sym_21_upper\":22,\"sym_22_lower\":22,\"sym_22_upper\":23,\"sym_23_lower\":23,\"sym_23_upper\":24,\"sym_24_lower\":24,\"sym_24_upper\":25,\"sym_25_lower\":25,\"sym_25_upper\":26,\"sym_26_lower\":26,\"sym_26_upper\":27,\"sym_27_lower\":27,\"sym_27_upper\":28,\"sym_28_lower\":28,\"sym_28_upper\":29,\"sym_29_lower\":29,\"sym_29_upper\":30,\"sym_2_lower\":2,\"sym_2_upper\":3,\"sym_30_lower\":30,\"sym_30_upper\":31,\"sym_31_lower\":31,\"sym_31_upper\":32,\"sym_32_lower\":32,\"sym_32_upper\":33,\"sym_33_lower\":33,\"sym_33_upper\":34,\"sym_34_lower\":34,\"sym_34_upper\":35,\"sym_35_lower\":35,\"sym_35_upper\":36,\"sym_36_lower\":36,\"sym_36_upper\":37,\"sym_37_lower\":37,\"sym_37_upper\":38,\"sym_38_lower\":38,\"sym_38_upper\":39,\"sym_39_lower\":39,\"sym_39_upper\":40,\"sym_3_lower\":3,\"sym_3_upper\":4,\"sym_40_lower\":40,\"sym_40_upper\":41,\"sym_41_lower\":41,\"sym_41_upper\":43,\"sym_42_lower\":43,\"sym_42_upper\":44,\"sym_43_lower\":44,\"sym_43_upper\":47,\"sym_44_lower\":47,\"sym_44_upper\":50,\"sym_45_lower\":50,\"sym_45_upper\":52,\"sym_46_lower\":52,\"sym_46_upper\":54,\"sym_47_lower\":54,\"sym_47_upper\":61,\"sym_48_lower\":61,\"sym_48_upper\":70,\"sym_49_lower\":70,\"sym_49_upper\":77,\"sym_4_lower\":4,\"sym_4_upper\":5,\"sym_50_lower\":77,\"sym_50_upper\":90,\"sym_51_lower\":90,\"sym_51_upper\":104,\"sym_52_lower\":104,\"sym_52_upper\":125,\"sym_53_lower\":125,\"sym_53_upper\":142,\"sym_54_lower\":142,\"sym_54_upper\":171,\"sym_55_lower\":171,\"sym_55_upper\":216,\"sym_56_lower\":216,\"sym_56_upper\":266,\"sym_57_lower\":266,\"sym_57_upper\":335,\"sym_58_lower\":335,\"sym_58_upper\":416,\"sym_59_lower\":416,\"sym_59_upper\":572,\"sym_5_lower\":5,\"sym_5_upper\":6,\"sym_60_lower\":572,\"sym_60_upper\":792,\"sym_61_lower\":792,\"sym_61_upper\":1167,\"sym_62_lower\":1167,\"sym_62_upper\":1950,\"sym_63_lower\":1950,\"sym_63_upper\":4265,\"sym_64_lower\":4265,\"sym_64_upper\":28851,\"sym_65_lower\":28851,\"sym_65_upper\":31167,\"sym_66_lower\":31167,\"sym_66_upper\":31950,\"sym_67_lower\":31950,\"sym_67_upper\":32294,\"sym_68_lower\":32294,\"sym_68_upper\":32483,\"sym_69_lower\":32483,\"sym_69_upper\":32609,\"sym_6_lower\":6,\"sym_6_upper\":7,\"sym_70_lower\":32609,\"sym_70_upper\":32670,\"sym_71_lower\":32670,\"sym_71_upper\":32720,\"sym_72_lower\":32720,\"sym_72_upper\":32761,\"sym_73_lower\":32761,\"sym_73_upper\":32782,\"sym_74_lower\":32782,\"sym_74_upper\":32804,\"sym_75_lower\":32804,\"sym_75_upper\":32810,\"sym_76_lower\":32810,\"sym_76_upper\":32815,\"sym_77_lower\":32815,\"sym_77_upper\":32825,\"sym_78_lower\":32825,\"sym_78_upper\":32832,\"sym_79_lower\":32832,\"sym_79_upper\":32837,\"sym_7_lower\":7,\"sym_7_upper\":8,\"sym_80_lower\":32837,\"sym_80_upper\":32840,\"sym_81_lower\":32840,\"sym_81_upper\":32844,\"sym_82_lower\":32844,\"sym_82_upper\":32847,\"sym_83_lower\":32847,\"sym_83_upper\":32849,\"sym_84_lower\":32849,\"sym_84_upper\":32850,\"sym_85_lower\":32850,\"sym_85_upper\":32852,\"sym_86_lower\":32852,\"sym_86_upper\":32854,\"sym_87_lower\":32854,\"sym_87_upper\":32856,\"sym_88_lower\":32856,\"sym_88_upper\":32857,\"sym_89_lower\":32857,\"sym_89_upper\":32858,\"sym_8_lower\":8,\"sym_8_upper\":9,\"sym_90_lower\":32858,\"sym_90_upper\":32859,\"sym_91_lower\":32859,\"sym_91_upper\":32860,\"sym_92_lower\":32860,\"sym_92_upper\":32861,\"sym_93_lower\":32861,\"sym_93_upper\":32862,\"sym_94_lower\":32862,\"sym_94_upper\":32863,\"sym_95_lower\":32863,\"sym_95_upper\":32864,\"sym_96_lower\":32864,\"sym_96_upper\":32865,\"sym_97_lower\":32865,\"sym_97_upper\":32866,\"sym_98_lower\":32866,\"sym_98_upper\":32867,\"sym_99_lower\":32867,\"sym_99_upper\":32868,\"sym_9_lower\":9,\"sym_9_upper\":10,\"sym_end_lower\":32896,\"sym_end_upper\":32898}" \
   ); */
  codec::ArithmeticEncoder<codec::SimpleAdaptiveModel> encoder(DCT_MAX - \
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
            // << std::chrono::duration_cast<std::chrono::duration<double>>(
            //        encode_t2 - encode_t1)
            //        .count()
            // << std::endl;

  std::vector<char> encoding = encoder.finish();
  //std::cout << encoder.dump_model() << std::endl;
  
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
            // << std::chrono::duration_cast<std::chrono::duration<double>>(
            //        serialize_t2 - serialize_t1)
            //        .count()
            // << std::endl;

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

  codec::ArithmeticDecoder<codec::SimpleAdaptiveModel> decoder(encoding_, DCT_MAX - DCT_MIN + 1);
  
  /*  codec::ArithmeticDecoder<codec::SimpleAdaptiveModel> decoder( encoding_, \
   "{\"denominator\":32899,\"num_symbols\":128,\"sym_0_lower\":0,\"sym_0_upper\":1,\"sym_100_lower\":32868,\"sym_100_upper\":32869,\"sym_101_lower\":32869,\"sym_101_upper\":32870,\"sym_102_lower\":32870,\"sym_102_upper\":32871,\"sym_103_lower\":32871,\"sym_103_upper\":32872,\"sym_104_lower\":32872,\"sym_104_upper\":32873,\"sym_105_lower\":32873,\"sym_105_upper\":32874,\"sym_106_lower\":32874,\"sym_106_upper\":32875,\"sym_107_lower\":32875,\"sym_107_upper\":32876,\"sym_108_lower\":32876,\"sym_108_upper\":32877,\"sym_109_lower\":32877,\"sym_109_upper\":32878,\"sym_10_lower\":10,\"sym_10_upper\":11,\"sym_110_lower\":32878,\"sym_110_upper\":32879,\"sym_111_lower\":32879,\"sym_111_upper\":32880,\"sym_112_lower\":32880,\"sym_112_upper\":32881,\"sym_113_lower\":32881,\"sym_113_upper\":32882,\"sym_114_lower\":32882,\"sym_114_upper\":32883,\"sym_115_lower\":32883,\"sym_115_upper\":32884,\"sym_116_lower\":32884,\"sym_116_upper\":32885,\"sym_117_lower\":32885,\"sym_117_upper\":32886,\"sym_118_lower\":32886,\"sym_118_upper\":32887,\"sym_119_lower\":32887,\"sym_119_upper\":32888,\"sym_11_lower\":11,\"sym_11_upper\":12,\"sym_120_lower\":32888,\"sym_120_upper\":32889,\"sym_121_lower\":32889,\"sym_121_upper\":32890,\"sym_122_lower\":32890,\"sym_122_upper\":32891,\"sym_123_lower\":32891,\"sym_123_upper\":32892,\"sym_124_lower\":32892,\"sym_124_upper\":32893,\"sym_125_lower\":32893,\"sym_125_upper\":32894,\"sym_126_lower\":32894,\"sym_126_upper\":32895,\"sym_127_lower\":32895,\"sym_127_upper\":32896,\"sym_12_lower\":12,\"sym_12_upper\":13,\"sym_13_lower\":13,\"sym_13_upper\":14,\"sym_14_lower\":14,\"sym_14_upper\":15,\"sym_15_lower\":15,\"sym_15_upper\":16,\"sym_16_lower\":16,\"sym_16_upper\":17,\"sym_17_lower\":17,\"sym_17_upper\":18,\"sym_18_lower\":18,\"sym_18_upper\":19,\"sym_19_lower\":19,\"sym_19_upper\":20,\"sym_1_lower\":1,\"sym_1_upper\":2,\"sym_20_lower\":20,\"sym_20_upper\":21,\"sym_21_lower\":21,\"sym_21_upper\":22,\"sym_22_lower\":22,\"sym_22_upper\":23,\"sym_23_lower\":23,\"sym_23_upper\":24,\"sym_24_lower\":24,\"sym_24_upper\":25,\"sym_25_lower\":25,\"sym_25_upper\":26,\"sym_26_lower\":26,\"sym_26_upper\":27,\"sym_27_lower\":27,\"sym_27_upper\":28,\"sym_28_lower\":28,\"sym_28_upper\":29,\"sym_29_lower\":29,\"sym_29_upper\":30,\"sym_2_lower\":2,\"sym_2_upper\":3,\"sym_30_lower\":30,\"sym_30_upper\":31,\"sym_31_lower\":31,\"sym_31_upper\":32,\"sym_32_lower\":32,\"sym_32_upper\":33,\"sym_33_lower\":33,\"sym_33_upper\":34,\"sym_34_lower\":34,\"sym_34_upper\":35,\"sym_35_lower\":35,\"sym_35_upper\":36,\"sym_36_lower\":36,\"sym_36_upper\":37,\"sym_37_lower\":37,\"sym_37_upper\":38,\"sym_38_lower\":38,\"sym_38_upper\":39,\"sym_39_lower\":39,\"sym_39_upper\":40,\"sym_3_lower\":3,\"sym_3_upper\":4,\"sym_40_lower\":40,\"sym_40_upper\":41,\"sym_41_lower\":41,\"sym_41_upper\":43,\"sym_42_lower\":43,\"sym_42_upper\":44,\"sym_43_lower\":44,\"sym_43_upper\":47,\"sym_44_lower\":47,\"sym_44_upper\":50,\"sym_45_lower\":50,\"sym_45_upper\":52,\"sym_46_lower\":52,\"sym_46_upper\":54,\"sym_47_lower\":54,\"sym_47_upper\":61,\"sym_48_lower\":61,\"sym_48_upper\":70,\"sym_49_lower\":70,\"sym_49_upper\":77,\"sym_4_lower\":4,\"sym_4_upper\":5,\"sym_50_lower\":77,\"sym_50_upper\":90,\"sym_51_lower\":90,\"sym_51_upper\":104,\"sym_52_lower\":104,\"sym_52_upper\":125,\"sym_53_lower\":125,\"sym_53_upper\":142,\"sym_54_lower\":142,\"sym_54_upper\":171,\"sym_55_lower\":171,\"sym_55_upper\":216,\"sym_56_lower\":216,\"sym_56_upper\":266,\"sym_57_lower\":266,\"sym_57_upper\":335,\"sym_58_lower\":335,\"sym_58_upper\":416,\"sym_59_lower\":416,\"sym_59_upper\":572,\"sym_5_lower\":5,\"sym_5_upper\":6,\"sym_60_lower\":572,\"sym_60_upper\":792,\"sym_61_lower\":792,\"sym_61_upper\":1167,\"sym_62_lower\":1167,\"sym_62_upper\":1950,\"sym_63_lower\":1950,\"sym_63_upper\":4265,\"sym_64_lower\":4265,\"sym_64_upper\":28851,\"sym_65_lower\":28851,\"sym_65_upper\":31167,\"sym_66_lower\":31167,\"sym_66_upper\":31950,\"sym_67_lower\":31950,\"sym_67_upper\":32294,\"sym_68_lower\":32294,\"sym_68_upper\":32483,\"sym_69_lower\":32483,\"sym_69_upper\":32609,\"sym_6_lower\":6,\"sym_6_upper\":7,\"sym_70_lower\":32609,\"sym_70_upper\":32670,\"sym_71_lower\":32670,\"sym_71_upper\":32720,\"sym_72_lower\":32720,\"sym_72_upper\":32761,\"sym_73_lower\":32761,\"sym_73_upper\":32782,\"sym_74_lower\":32782,\"sym_74_upper\":32804,\"sym_75_lower\":32804,\"sym_75_upper\":32810,\"sym_76_lower\":32810,\"sym_76_upper\":32815,\"sym_77_lower\":32815,\"sym_77_upper\":32825,\"sym_78_lower\":32825,\"sym_78_upper\":32832,\"sym_79_lower\":32832,\"sym_79_upper\":32837,\"sym_7_lower\":7,\"sym_7_upper\":8,\"sym_80_lower\":32837,\"sym_80_upper\":32840,\"sym_81_lower\":32840,\"sym_81_upper\":32844,\"sym_82_lower\":32844,\"sym_82_upper\":32847,\"sym_83_lower\":32847,\"sym_83_upper\":32849,\"sym_84_lower\":32849,\"sym_84_upper\":32850,\"sym_85_lower\":32850,\"sym_85_upper\":32852,\"sym_86_lower\":32852,\"sym_86_upper\":32854,\"sym_87_lower\":32854,\"sym_87_upper\":32856,\"sym_88_lower\":32856,\"sym_88_upper\":32857,\"sym_89_lower\":32857,\"sym_89_upper\":32858,\"sym_8_lower\":8,\"sym_8_upper\":9,\"sym_90_lower\":32858,\"sym_90_upper\":32859,\"sym_91_lower\":32859,\"sym_91_upper\":32860,\"sym_92_lower\":32860,\"sym_92_upper\":32861,\"sym_93_lower\":32861,\"sym_93_upper\":32862,\"sym_94_lower\":32862,\"sym_94_upper\":32863,\"sym_95_lower\":32863,\"sym_95_upper\":32864,\"sym_96_lower\":32864,\"sym_96_upper\":32865,\"sym_97_lower\":32865,\"sym_97_upper\":32866,\"sym_98_lower\":32866,\"sym_98_upper\":32867,\"sym_99_lower\":32867,\"sym_99_upper\":32868,\"sym_9_lower\":9,\"sym_9_upper\":10,\"sym_end_lower\":32896,\"sym_end_upper\":32898}" \
   ); */
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
