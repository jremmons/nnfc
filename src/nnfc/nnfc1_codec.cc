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

using namespace std;

std::vector<float> kmeans(nn::Tensor<float, 3> input, int nbins,
                          int max_iter = 10) {
  const int dim0 = input.dimension(0);
  const int dim1 = input.dimension(1);
  const int dim2 = input.dimension(2);
  std::vector<float> vals(&input(0, 0, 0),
                          &input(0, 0, 0) + dim0 * dim1 * dim2);
  std::sort(vals.begin(), vals.end());

  const float min = vals[0]; // min is first element since array is sorted.
  const float max = vals[dim0 * dim1 * dim2 - 1]; // max is last element since array is sorted. 

  // initial means to be linearly spaced
  std::vector<float> means(nbins);
  for (int i = 0; i < nbins; i++) {
    means[i] = min + ((max - min) * (static_cast<float>(i) / (nbins - 1)));
  }

  // perform multiple iterations of llyod's algorithm
  for (int iter = 0; iter < max_iter; iter++) {
      // TODO(jemmons) add k-mean here. 
  }

  return means;
}

int quantize(float val, std::vector<float> &bins) {
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

vector<uint8_t> nnfc::NNFC1Encoder::forward(nn::Tensor<float, 3> input) {
  uint64_t dim0 = input.dimension(0);
  uint64_t dim1 = input.dimension(1);
  uint64_t dim2 = input.dimension(2);

  // quantize the input data
  std::vector<float> means = kmeans(input, 256);
  for (size_t i = 0; i < dim0; i++) {
    for (size_t j = 0; j < dim1; j++) {
      for (size_t k = 0; k < dim2; k++) {
        input(i, j, k) = means[quantize(input(i, j, k), means)];
      }
    }
  }

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

  // const uint64_t dim0 = input.dimension(0);
  // const uint64_t dim1 = input.dimension(1);
  // const uint64_t dim2 = input.dimension(2);

  // const float min = input.minimum();
  // const float max = input.maximum();

  // // create a square grid for the activations to go into
  // const size_t jpeg_chunks = ceil(sqrt(dim0));
  // const size_t jpeg_height = jpeg_chunks * dim1;
  // const size_t jpeg_width = jpeg_chunks * dim2;

  // vector<uint8_t> buffer(jpeg_height * jpeg_width + 1024);
  // fill(buffer.begin(), buffer.end(), 0);

  // // compute the strides for laying out the data in memory
  // const size_t row_channel_stride = dim1 * dim2;
  // const size_t row_stride = jpeg_chunks * dim2;
  // const size_t channel_stride = dim2;

  // Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> input_q =
  //   ((input.tensor() - min) * (255 / (max - min))).cast<uint8_t>();

  // // swizzle the data into the right memory layout
  // for(size_t row_channel = 0;
  //     row_channel < jpeg_chunks * jpeg_chunks;
  //     row_channel += jpeg_chunks) {
  //     for(size_t row = 0; row < dim1; row++) {
  //         for(size_t channel = 0; channel < jpeg_chunks; channel++) {
  //             if(row_channel + channel < dim0) {
  //                 const size_t offset = row_channel_stride * row_channel +
  //                                       row_stride * row +
  //                                       channel_stride * channel;
  //                 memcpy(&buffer[offset], &input_q(row_channel + channel,
  //                 row, 0), dim2);
  //             }
  //         }
  //     }
  // }

  // // jpeg compress the data
  // long unsigned int jpeg_size = 0;
  // unsigned char* compressed_image = nullptr;
  // tjCompress2(jpeg_compressor.get(), buffer.data(), jpeg_width, 0,
  // jpeg_height, TJPF_GRAY,
  //             &compressed_image, &jpeg_size, TJSAMP_GRAY, quantizer_,
  //             TJFLAG_FASTDCT);

  // // serialize
  // vector<uint8_t> encoding(jpeg_size);
  // memcpy(encoding.data(), compressed_image, jpeg_size);
  // tjFree(compressed_image);

  // const uint8_t *min_bytes = reinterpret_cast<const uint8_t *>(&min);
  // const uint8_t *max_bytes = reinterpret_cast<const uint8_t *>(&max);

  // for(size_t i = 0; i < sizeof(float); i++){
  //     encoding.push_back(min_bytes[i]);
  // }
  // for(size_t i = 0; i < sizeof(float); i++){
  //     encoding.push_back(max_bytes[i]);
  // }

  // const uint8_t *dim0_bytes = reinterpret_cast<const uint8_t*>(&dim0);
  // const uint8_t *dim1_bytes = reinterpret_cast<const uint8_t*>(&dim1);
  // const uint8_t *dim2_bytes = reinterpret_cast<const uint8_t*>(&dim2);

  // for(size_t i = 0; i < sizeof(uint64_t); i++){
  //     encoding.push_back(dim0_bytes[i]);
  // }
  // for(size_t i = 0; i < sizeof(uint64_t); i++){
  //     encoding.push_back(dim1_bytes[i]);
  // }
  // for(size_t i = 0; i < sizeof(uint64_t); i++){
  //     encoding.push_back(dim2_bytes[i]);
  // }

  // return encoding;
}

nn::Tensor<float, 3> nnfc::NNFC1Encoder::backward(nn::Tensor<float, 3> input) {
  return input;
}

nnfc::NNFC1Decoder::NNFC1Decoder()
    : jpeg_decompressor(tjInitDecompress(), [](void *ptr) { tjDestroy(ptr); }) {
}

nnfc::NNFC1Decoder::~NNFC1Decoder() {}

nn::Tensor<float, 3> nnfc::NNFC1Decoder::forward(vector<uint8_t> input) {
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

  // uint64_t dim0;
  // uint64_t dim1;
  // uint64_t dim2;
  // uint8_t *dim0_bytes = reinterpret_cast<uint8_t*>(&dim0);
  // uint8_t *dim1_bytes = reinterpret_cast<uint8_t*>(&dim1);
  // uint8_t *dim2_bytes = reinterpret_cast<uint8_t*>(&dim2);

  // size_t length = input.size();
  // size_t dim0_offset = length - 3*sizeof(uint64_t);
  // size_t dim1_offset = length - 2*sizeof(uint64_t);
  // size_t dim2_offset = length - 1*sizeof(uint64_t);
  // for(size_t i = 0; i < sizeof(uint64_t); i++){
  //     dim0_bytes[i] = input[i + dim0_offset];
  //     dim1_bytes[i] = input[i + dim1_offset];
  //     dim2_bytes[i] = input[i + dim2_offset];
  // }

  // float min;
  // float max;
  // uint8_t *min_bytes = reinterpret_cast<uint8_t*>(&min);
  // uint8_t *max_bytes = reinterpret_cast<uint8_t*>(&max);
  // size_t min_offset = length - 3*sizeof(uint64_t) - 2*sizeof(float);
  // size_t max_offset = length - 3*sizeof(uint64_t) - 1*sizeof(float);
  // for(size_t i = 0; i < sizeof(uint64_t); i++){
  //     min_bytes[i] = input[i + min_offset];
  //     max_bytes[i] = input[i + max_offset];
  // }
  // // cout << min << " " << max << "\n";

  // const size_t jpeg_chunks = ceil(sqrt(dim0));
  // // const size_t jpeg_height = jpeg_chunks*dim1;
  // // const size_t jpeg_width = jpeg_chunks*dim2;

  // const long unsigned int jpeg_size = input.size() - 3*sizeof(uint64_t) -
  // 2*sizeof(float);

  // int jpegSubsamp, width, height;
  // tjDecompressHeader2(jpeg_decompressor.get(), input.data(), jpeg_size,
  // &width, &height, &jpegSubsamp);

  // // cout << jpeg_height << " " << height << "\n";
  // // cout << jpeg_width << " " << width << "\n";

  // vector<uint8_t> buffer(width * height);

  // tjDecompress2(jpeg_decompressor.get(), input.data(), jpeg_size,
  // buffer.data(), width, 0/*pitch*/, height, TJPF_GRAY, TJFLAG_FASTDCT);

  // // compute the strides for laying out the data in memory
  // const size_t row_channel_stride = dim1 * dim2;
  // const size_t row_stride = jpeg_chunks * dim2;
  // const size_t channel_stride = dim2;
  // const size_t col_stride = 1;

  // // swizzle the data into the right memory layout
  // nn::Tensor<float, 3> output(dim0, dim1, dim2);

  // for(size_t row_channel = 0; row_channel < jpeg_chunks*jpeg_chunks;
  // row_channel += jpeg_chunks) {

  //     for(size_t row = 0; row < dim1; row++) {
  //         for(size_t channel = 0; channel < jpeg_chunks; channel++) {

  //             if(row_channel + channel < dim0) {
  //                 for(size_t col = 0; col < dim2; col++) {

  //                     const size_t offset = row_channel_stride*row_channel +
  //                                           row_stride*row +
  //                                           channel_stride*channel +
  //                                           col_stride*col;
  //                     const double val = buffer[offset];
  //                     output(row_channel + channel, row, col) =
  //                     static_cast<float>((max-min) * (val/255) + min);
  //                 }
  //             }
  //         }
  //     }
  // }

  // return output;
}

nn::Tensor<float, 3> nnfc::NNFC1Decoder::backward(nn::Tensor<float, 3> input) {
  return input;
}
