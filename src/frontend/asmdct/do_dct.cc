#include <bitset>
#include <cmath>
#include <iomanip>
#include <iostream>

//#include "../codec/tjdct/jsimd.hh"
#include "../codec/fastdct.hh"
//#include "../codec/utils.hh"
#include "../nn/tensor.hh"

// alignas(16) static int16_t buffer[64] = {0};
// alignas(16) static int16_t coef[64] = {0};

// alignas(16) static uint8_t out[64] = {0};
// alignas(16) static uint8_t *out_buffer[8];

// static int16_t dct_table[64] = {0};

int main() {
  // out_buffer[0] = &out[0];
  // out_buffer[1] = &out[8];
  // out_buffer[2] = &out[16];
  // out_buffer[3] = &out[24];
  // out_buffer[4] = &out[32];
  // out_buffer[5] = &out[40];
  // out_buffer[6] = &out[48];
  // out_buffer[7] = &out[56];

  // for (int i = 0; i < 8; i++) {
  //   for (int j = 0; j < 8; j++) {
  //     buffer[8 * i + j] = 1;
  //     std::cout << std::setfill(' ') << std::setw(4) << buffer[8 * i + j]
  //               << " ";
  //   }
  //   std::cout << "\n";
  // }
  // std::cout << "\n";

  // jsimd_fdct_ifast_sse2(buffer);

  // for (int i = 0; i < 8; i++) {
  //   for (int j = 0; j < 8; j++) {
  //     std::cout << std::setfill(' ') << std::setw(4) << buffer[8 * i + j]
  //               << " ";
  //   }
  //   std::cout << "\n";
  // }
  // std::cout << "\n";

  // for (int i = 0; i < 8; i++) {
  //   for (int j = 0; j < 8; j++) {
  //     coef[8 * i + j] = 42;
  //   }
  // }

  // //jsimd_quantize_sse2(coef, divisors, buffer);
  // jsimd_quantize_sse2(buffer, divisors, buffer);

  // for (int i = 0; i < 8; i++) {
  //   for (int j = 0; j < 8; j++) {
  //     //std::cout << std::setfill(' ') << std::setw(4) << coef[8 * i + j] << " ";
  //     std::cout << std::setfill(' ') << std::setw(4) << buffer[8 * i + j] << " ";
  //   }
  //   std::cout << "\n";
  // }
  // std::cout << "\n";

  // // for (int i = 0; i < 64; i++) {
  // //     std::cout << std::setfill(' ') << std::setw(4) << divisors[i] << " ";
  // //     std::cout << "\n";
  // // }
  // // std::cout << "\n";

  // jsimd_idct_ifast_sse2(dct_table, buffer,
  //                       reinterpret_cast<unsigned char **>(out_buffer), 0);

  // for (int i = 0; i < 8; i++) {
  //   for (int j = 0; j < 8; j++) {
  //     std::cout << std::setfill(' ') << std::setw(4)
  //               << (int)out[8 * i + j] - 128 << " ";
  //   }
  //   std::cout << "\n";
  // }
  // std::cout << "\n";

  // for (int i = 0; i < 8; i++) {
  //     for (int j = 0; j < 8; j++) {
  //         std::cout << std::setfill(' ') << std::setw(3) <<
  //         std::bitset<16>(out[8*i + j])  << " ";
  //     }
  //     std::cout << "\n";
  // }
  // std::cout << "\n";

  nn::Tensor<int16_t, 3> input(1,8,8);
  codec::FastDCT dct;
  codec::FastIDCT idct;
  
  for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
          input(0,i,j) = (255 / 15) * (i + j);
          std::cout <<  std::setfill(' ') << std::setw(4) << input(0,i,j) <<
          " ";
      }
      std::cout << "\n";
  }
  std::cout << "\n";

  nn::Tensor<int16_t, 3> dct_output = dct(input);
  //nn::Tensor<float, 3> dct = codec::utils::dct(input);

  for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
          std::cout << std::setfill(' ') << std::setw(4) << dct_output(0,i,j) << " ";
      }
      std::cout << "\n";
  }
  std::cout << "\n";

  nn::Tensor<uint8_t, 3> idct_output = idct(dct_output);
  //nn::Tensor<float, 3> idct = codec::utils::idct(dct);

  for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
          std::cout << std::setfill(' ') << std::setw(4) << (int)idct_output(0,i,j) << " ";
      }
      std::cout << "\n";
  }
  std::cout << "\n";


  std::cout << "diff \n";
  for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
          input(0,i,j) = (255 / 15) * (i + j);
          std::cout <<  std::setfill(' ') << std::setw(4) << input(0,i,j) - (int)idct_output(0,i,j) <<
          " ";
      }
      std::cout << "\n";
  }
  std::cout << "\n";

  
  return 0;
}
