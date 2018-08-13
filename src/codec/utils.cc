#include "utils.hh"

#include <iostream>
#include <stdexcept>

extern "C" {
#include <fftw3.h>
}

using namespace std;

nn::Tensor<float, 3> _dct_idct_f32(nn::Tensor<float, 3>& input, const int N,
                               bool inverse) {
  const int channels = input.dimension(0);
  const int rows = input.dimension(1);
  const int cols = input.dimension(2);

  if ((rows % N) or (cols % N)) {
    throw runtime_error("rows and cols must be multiples of N");
  }

  const int block_rows = rows / N;
  const int block_cols = cols / N;

  nn::Tensor<float, 3> output(channels, rows, cols);

  for (int channel = 0; channel < channels; channel++) {
    for (int col = 0; col < block_cols; col++) {
      const int ranks[] = {N, N};
      const int nembed[] = {rows, cols};
      const int howmany = block_rows;
      const int stride = 1;
      const int dist = (N * cols);
      float* input_start = &input(channel, 0, 0) + col * N;
      float* output_start = &output(channel, 0, 0) + col * N;
      const fftw_r2r_kind kinds[] = {inverse ? FFTW_REDFT01 : FFTW_REDFT10,
                                     inverse ? FFTW_REDFT01 : FFTW_REDFT10};

      fftwf_plan plan = fftwf_plan_many_r2r(
          2, ranks, howmany, input_start, nembed, stride, dist, output_start,
          nembed, stride, dist, kinds, FFTW_ESTIMATE);

      fftwf_execute(plan);
      fftwf_destroy_plan(plan);
    }
  }

  if (not inverse) {
    const float scale_factor = 1.0 / (4 * N * N);
    for (int i = 0; i < channels; i++) {
      for (int j = 0; j < rows; j++) {
        for (int k = 0; k < rows; k++) {
          output(i, j, k) *= scale_factor;
        }
      }
    }
  }

  return output;
}

// nn::Tensor<float, 3> codec::utils::dct(nn::Tensor<float, 3>& input,
//                                        const int N) {
//   return _dct_idct_f32(input, N, false);
// }

// nn::Tensor<float, 3> codec::utils::idct(nn::Tensor<float, 3>& input,
//                                         const int N) {
//   return _dct_idct_f32(input, N, true);
// }


// nn::Tensor<uint8_t, 3> _dct_idct_u8(nn::Tensor<uint8_t, 3>& input, const int N,
//                                bool inverse) {
//   const int channels = input.dimension(0);
//   const int rows = input.dimension(1);
//   const int cols = input.dimension(2);

//   if ((rows % N) or (cols % N)) {
//     throw runtime_error("rows and cols must be multiples of N");
//   }

//   const int block_rows = rows / N;
//   const int block_cols = cols / N;

//   nn::Tensor<uint8_t, 3> output(channels, rows, cols);

//   for (int channel = 0; channel < channels; channel++) {
//     for (int col = 0; col < block_cols; col++) {
//       const int ranks[] = {N, N};
//       const int nembed[] = {rows, cols};
//       const int howmany = block_rows;
//       const int stride = 1;
//       const int dist = (N * cols);
//       uint8_t* input_start = &input(channel, 0, 0) + col * N;
//       uint8_t* output_start = &output(channel, 0, 0) + col * N;
//       const fftw_r2r_kind kinds[] = {inverse ? FFTW_REDFT01 : FFTW_REDFT10,
//                                      inverse ? FFTW_REDFT01 : FFTW_REDFT10};

//       fftwf_plan plan = fftwf_plan_many_r2r(
//           2, ranks, howmany, input_start, nembed, stride, dist, output_start,
//           nembed, stride, dist, kinds, FFTW_ESTIMATE);

//       fftwf_execute(plan);
//       fftwf_destroy_plan(plan);
//     }
//   }

//   if (not inverse) {
//     const float scale_factor = 1.0 / (4 * N * N);
//     for (int i = 0; i < channels; i++) {
//       for (int j = 0; j < rows; j++) {
//         for (int k = 0; k < rows; k++) {
//           output(i, j, k) *= scale_factor;
//         }
//       }
//     }
//   }

//   return output;
// }
