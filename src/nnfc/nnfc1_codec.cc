#include <turbojpeg.h>

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "codec/utils.hh"
#include "nnfc1_codec.hh"
#include "tensor.hh"

using namespace std;

std::vector<float> kmeans(nn::Tensor<float, 3> input, int nbins,
                          int max_iter = 100) {
  const int dim0 = input.dimension(0);
  const int dim1 = input.dimension(1);
  const int dim2 = input.dimension(2);
  std::vector<float> vals(&input(0, 0, 0),
                          &input(0, 0, 0) + dim0 * dim1 * dim2);
  const int vals_size = vals.size();

  // use kmeans++ initialization strategy
  assert(nbins > 1);
  std::vector<float> means(nbins);

  // initialize the means
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<float> distances_squared(vals_size);

    // set the initial mean and distances
    {
      std::uniform_int_distribution<> dist(0, vals_size - 1);
      means[0] = vals[dist(gen)];

      for (int i = 0; i < vals_size; i++) {
        const float val = vals[i];
        distances_squared[i] = (means[0] - val) * (means[0] - val);
      }
    }

    // set the other nbins-1 means
    {
      for (int bin = 1; bin < nbins; bin++) {
        std::discrete_distribution<> d(distances_squared.begin(),
                                       distances_squared.end());

        const int new_mean_idx = d(gen);
        means[bin] = vals[new_mean_idx];

        for (int i = 0; i < vals_size; i++) {
          const float val = vals[i];
          const float curr_distance_squared =
              (means[bin] - val) * (means[bin] - val);
          if (curr_distance_squared < distances_squared[i]) {
            distances_squared[i] = curr_distance_squared;
          }
        }
      }
    }
  }

  // perform kmeans `max_iter` times
  std::vector<float> means_sum(nbins);
  std::vector<float> means_count(nbins);

  for (int iter = 0; iter < max_iter; iter++) {
    for (int i = 0; i < nbins; i++) {
      means_count[i] = 0;
      means_sum[i] = 0;
    }

    for (int i = 0; i < vals_size; i++) {
      const float val = vals[i];

      int best_mean = -1;
      float best_distance = FLT_MAX;
      for (int j = 0; j < nbins; j++) {
        const float curr_distance = (means[j] - val) * (means[j] - val);
        if (curr_distance < best_distance) {
          best_distance = curr_distance;
          best_mean = j;
        }
      }

      assert(best_mean >= 0);
      means_count[best_mean] += 1;
      means_sum[best_mean] += val;
    }
  }

  std::sort(means.begin(), means.end());
  return means;
}

uint32_t quantize(float val, std::vector<float> &bins) {
  int lower = 0;
  int upper = bins.size() - 1;

  while (upper - lower > 1) {
    int mid = (lower + upper) / 2;

    if (bins[mid] < val) {
      lower = mid;
    } else {
      upper = mid;
    }
  }

  float lower_error = val - bins[lower];
  float upper_error = bins[upper] - val;
  if (lower_error < upper_error) {
    return lower;
  } else {
    return upper;
  }
}

nnfc::NNFC1Encoder::NNFC1Encoder() : quantizer_nbins_(4) {}

nnfc::NNFC1Encoder::~NNFC1Encoder() {}

constexpr int ZIGZAG_ORDER[][2] = {
    {0, 0}, {0, 1}, {1, 0}, {2, 0}, {1, 1}, {0, 2}, {0, 3}, {1, 2},
    {2, 1}, {3, 0}, {3, 1}, {2, 2}, {1, 3}, {2, 3}, {3, 2}, {3, 3},
};

vector<uint8_t> nnfc::NNFC1Encoder::forward(nn::Tensor<float, 3> t_input) {
  nn::Tensor<float, 3> input(move(codec::utils::dct(t_input, BLOCK_WIDTH)));

  uint64_t dim0 = input.dimension(0);
  uint64_t dim1 = input.dimension(1);
  uint64_t dim2 = input.dimension(2);

  assert(__builtin_popcount(quantizer_nbins_) == 1);
  assert(quantizer_nbins_ < 256);
  std::vector<float> means = kmeans(input, quantizer_nbins_);

  // quantize the input data
  uint32_t count = 0;
  uint8_t byte = 0;
  std::vector<uint8_t> encoding;

  const uint64_t block_rows = dim1 / BLOCK_WIDTH;
  const uint64_t block_cols = dim2 / BLOCK_WIDTH;

  for (size_t i = 0; i < dim0; i++) { /* channels */
    for (size_t jj = 0; jj < block_rows; jj++) {
      for (size_t kk = 0; kk < block_cols; kk++) {
        for (size_t zz = 0; zz < BLOCK_WIDTH * BLOCK_WIDTH; zz++) {
          const size_t j = ZIGZAG_ORDER[zz][0];
          const size_t k = ZIGZAG_ORDER[zz][1];

          if (count % 4 == 0 and count != 0) {
            encoding.push_back(byte);
            byte = 0;
          }

          const float val =
              input(i, jj * BLOCK_WIDTH + j, kk * BLOCK_WIDTH + k);
          const uint32_t qval = quantize(val, means);
          assert(qval <= 0b11);
          const uint32_t shift = 2 * (count % 4);
          byte |= (static_cast<uint8_t>(qval) << shift);
          count++;
        }
      }
    }
  }
  encoding.push_back(byte);  // push the last byte

  {
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
  }

  {
    uint32_t nbins = quantizer_nbins_;
    uint8_t *nbins_bytes = reinterpret_cast<uint8_t *>(&nbins);
    for (size_t i = 0; i < sizeof(uint32_t); i++) {
      encoding.push_back(nbins_bytes[i]);
    }

    for (int bin = 0; bin < quantizer_nbins_; bin++) {
      uint8_t *bin_bytes = reinterpret_cast<uint8_t *>(&means[bin]);
      for (size_t i = 0; i < sizeof(float); i++) {
        encoding.push_back(bin_bytes[i]);
      }
    }
  }

  return encoding;
}

nn::Tensor<float, 3> nnfc::NNFC1Encoder::backward(nn::Tensor<float, 3> input) {
  return input;
}

nnfc::NNFC1Decoder::NNFC1Decoder() {}

nnfc::NNFC1Decoder::~NNFC1Decoder() {}

nn::Tensor<float, 3> nnfc::NNFC1Decoder::forward(vector<uint8_t> input) {
  const size_t length = input.size();

  uint64_t dim0;
  uint64_t dim1;
  uint64_t dim2;
  {
    uint8_t *dim0_bytes = reinterpret_cast<uint8_t *>(&dim0);
    uint8_t *dim1_bytes = reinterpret_cast<uint8_t *>(&dim1);
    uint8_t *dim2_bytes = reinterpret_cast<uint8_t *>(&dim2);

    size_t dim0_offset = length - 3 * sizeof(uint64_t) - 4 * sizeof(float) -
                         1 * sizeof(uint32_t);
    size_t dim1_offset = length - 2 * sizeof(uint64_t) - 4 * sizeof(float) -
                         1 * sizeof(uint32_t);
    size_t dim2_offset = length - 1 * sizeof(uint64_t) - 4 * sizeof(float) -
                         1 * sizeof(uint32_t);
    for (size_t i = 0; i < sizeof(uint64_t); i++) {
      dim0_bytes[i] = input[i + dim0_offset];
      dim1_bytes[i] = input[i + dim1_offset];
      dim2_bytes[i] = input[i + dim2_offset];
    }
  }

  uint32_t nbins;
  {
    uint8_t *nbins_bytes = reinterpret_cast<uint8_t *>(&nbins);
    size_t nbins_offset = length - 4 * sizeof(float) - 1 * sizeof(uint32_t);
    for (size_t i = 0; i < sizeof(uint32_t); i++) {
      nbins_bytes[i] = input[i + nbins_offset];
    }
  }
  assert(nbins < 256);

  std::vector<float> means;
  means.resize(nbins);
  {
    for (uint32_t bin = 0; bin < nbins; bin++) {
      size_t bin_offset = length - (nbins - bin) * sizeof(float);

      uint8_t *bin_bytes = reinterpret_cast<uint8_t *>(&means[bin]);
      for (size_t i = 0; i < sizeof(float); i++) {
        bin_bytes[i] = input[i + bin_offset];
      }
    }
  }

  nn::Tensor<float, 3> output(dim0, dim1, dim2);

  const uint64_t block_rows = dim1 / BLOCK_WIDTH;
  const uint64_t block_cols = dim2 / BLOCK_WIDTH;

  uint32_t count = 0;
  uint8_t byte = 0;

  for (size_t i = 0; i < dim0; i++) { /* channels */
    for (size_t jj = 0; jj < block_rows; jj++) {
      for (size_t kk = 0; kk < block_cols; kk++) {
        for (size_t zz = 0; zz < BLOCK_WIDTH * BLOCK_WIDTH; zz++) {
          const size_t j = ZIGZAG_ORDER[zz][0];
          const size_t k = ZIGZAG_ORDER[zz][1];

          if (count % 4 == 0) {
            byte = input[count >> 2];
          }

          uint8_t qval = (byte >> 2 * (count % 4)) & 0b11;
          output(i, jj * BLOCK_WIDTH + j, kk * BLOCK_WIDTH + k) = means[qval];
          count++;
        }
      }
    }
  }

  return codec::utils::idct(output, BLOCK_WIDTH);
}

nn::Tensor<float, 3> nnfc::NNFC1Decoder::backward(nn::Tensor<float, 3> input) {
  return input;
}
