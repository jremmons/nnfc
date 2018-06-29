#include <turbojpeg.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "nnfc1_codec.hh"
#include "tensor.hh"
#include "codec/utils.hh"

using namespace std;

std::vector<float> kmeans(nn::Tensor<float, 3> input, int nbins,
                          int max_iter = 10) {

  const int dim0 = input.dimension(0);
  const int dim1 = input.dimension(1);
  const int dim2 = input.dimension(2);
  std::vector<float> vals(&input(0, 0, 0),
                          &input(0, 0, 0) + dim0 * dim1 * dim2);

  const float min = input.minimum();
  const float max = input.maximum();

  // initial means to be linearly spaced
  assert(nbins > 1);
  std::vector<float> means(nbins);
  for (int i = 0; i < nbins; i++) {
    means[i] = min + ((max - min) * (static_cast<float>(i) / (nbins - 1)));
  }

  std::vector<float> means_sum(nbins);
  std::vector<float> means_count(nbins);

  for (int iter = 0; iter < max_iter; iter++) {

      for (int i = 0; i < nbins; i++) {
          means_count[i] = 0;
          means_sum[i] = 0;
      }

      const int vals_size = vals.size();
      for (int i = 0; i < vals_size; i++) {
          const float val = vals[i];

          int best_mean = 0;
          float best_distance = means[0] - val > 0 ? means[0] - val : val - means[0];
          for (int j = 1; j < nbins; j++) {
              const float curr_distance = means[j] - val > 0 ? means[j] - val : val - means[j];
              if (curr_distance < best_distance) {
                  best_distance = curr_distance;
                  best_mean = j;
              }
          }

          means_count[best_mean] += 1;
          means_sum[best_mean] += val;
      }

      for (int i = 0; i < nbins; i++) {
          means[i] = means_sum[i] / means_count[i];
      }

  }

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

nnfc::NNFC1Encoder::NNFC1Encoder(int quantizer)
    : quantizer_(quantizer),
      jpeg_compressor(tjInitCompress(), [](void *ptr) { tjDestroy(ptr); }) {}

nnfc::NNFC1Encoder::~NNFC1Encoder() {}

vector<uint8_t> nnfc::NNFC1Encoder::forward(nn::Tensor<float, 3> t_input) {
  nn::Tensor<float, 3> input(move(codec::utils::dct(t_input, 4)));

  uint64_t dim0 = input.dimension(0);
  uint64_t dim1 = input.dimension(1);
  uint64_t dim2 = input.dimension(2);

  std::vector<float> means = kmeans(input, 4);

  // quantize the input data
  uint32_t count = 0;
  uint8_t byte = 0;
  std::vector<uint8_t> encoding;
  for (size_t i = 0; i < dim0; i++) {
    for (size_t j = 0; j < dim1; j++) {
      for (size_t k = 0; k < dim2; k++) {

          if(count % 4 == 0 and count != 0) {
              encoding.push_back(byte);
              byte = 0;
          }
          const uint32_t qval = quantize(input(i, j, k), means);
          assert(qval <= 0b11);
          const uint32_t shift = 2 * (count % 4);
          byte |= (static_cast<uint8_t>(qval) << shift);
          count++;

      }
    }
  }
  encoding.push_back(byte); // push the last byte

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
    uint8_t *q0_bytes = reinterpret_cast<uint8_t *>(&means[0]);
    uint8_t *q1_bytes = reinterpret_cast<uint8_t *>(&means[1]);
    uint8_t *q2_bytes = reinterpret_cast<uint8_t *>(&means[2]);
    uint8_t *q3_bytes = reinterpret_cast<uint8_t *>(&means[3]);
    for (size_t i = 0; i < sizeof(float); i++) {
      encoding.push_back(q0_bytes[i]);
    }
    for (size_t i = 0; i < sizeof(float); i++) {
      encoding.push_back(q1_bytes[i]);
    }
    for (size_t i = 0; i < sizeof(float); i++) {
      encoding.push_back(q2_bytes[i]);
    }
    for (size_t i = 0; i < sizeof(float); i++) {
      encoding.push_back(q3_bytes[i]);
    }
  }

  return encoding;
}

nn::Tensor<float, 3> nnfc::NNFC1Encoder::backward(nn::Tensor<float, 3> input) {
  return input;
}

nnfc::NNFC1Decoder::NNFC1Decoder()
    : jpeg_decompressor(tjInitDecompress(), [](void *ptr) { tjDestroy(ptr); }) {
}

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

    size_t dim0_offset = length - 3*sizeof(uint64_t) - 4*sizeof(float);
    size_t dim1_offset = length - 2*sizeof(uint64_t) - 4*sizeof(float);
    size_t dim2_offset = length - 1*sizeof(uint64_t) - 4*sizeof(float);
    for (size_t i = 0; i < sizeof(uint64_t); i++) {
      dim0_bytes[i] = input[i + dim0_offset];
      dim1_bytes[i] = input[i + dim1_offset];
      dim2_bytes[i] = input[i + dim2_offset];
    }
  }

  float means[4];
  {
    uint8_t *q0_bytes = reinterpret_cast<uint8_t *>(&means[0]);
    uint8_t *q1_bytes = reinterpret_cast<uint8_t *>(&means[1]);
    uint8_t *q2_bytes = reinterpret_cast<uint8_t *>(&means[2]);
    uint8_t *q3_bytes = reinterpret_cast<uint8_t *>(&means[3]);

    size_t q0_offset = length - 4*sizeof(float);
    size_t q1_offset = length - 3*sizeof(float);
    size_t q2_offset = length - 2*sizeof(float);
    size_t q3_offset = length - 1*sizeof(float);
    for (size_t i = 0; i < sizeof(float); i++) {
        q0_bytes[i] = input[i + q0_offset];
        q1_bytes[i] = input[i + q1_offset];
        q2_bytes[i] = input[i + q2_offset];
        q3_bytes[i] = input[i + q3_offset];
    }
  }

  nn::Tensor<float, 3> output(dim0, dim1, dim2);

  uint32_t count = 0;
  uint8_t byte = 0;
  for (size_t i = 0; i < dim0; i++) {
    for (size_t j = 0; j < dim1; j++) {
      for (size_t k = 0; k < dim2; k++) {

          if(count % 4 == 0) {
              byte = input[count >> 2];
          }

          uint8_t qval = (byte >> 2*(count % 4)) & 0b11;
          output(i, j, k) = means[qval];
          count++;

      }
    }
  }

  return codec::utils::idct(output, 4);
}

nn::Tensor<float, 3> nnfc::NNFC1Decoder::backward(nn::Tensor<float, 3> input) {
  return input;
}
