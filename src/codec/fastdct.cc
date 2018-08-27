#include <cstdlib>
#include <memory>

#include "codec/tjdct/jsimd.hh"
#include "fastdct.hh"
#include "nn/tensor.hh"

static constexpr int alignment = 16;

inline std::unique_ptr<int16_t, void (*)(void *)> tj_data(
    const int size_, const int16_t *data_) {
  std::unique_ptr<int16_t, void (*)(void *)> data(
      static_cast<int16_t *>(std::aligned_alloc(alignment, size_)),
      [](void *ptr) { std::free(ptr); });
  std::memcpy(data.get(), data_, size_);

  return std::move(data);
}

codec::FastDCT::FastDCT()
    : divisors_(std::move(tj_data(sizeof(divisors), divisors))),
      work_buffer_(static_cast<int16_t *>(std::aligned_alloc(alignment, 128)),
                   [](void *ptr) { std::free(ptr); }) {}

codec::FastDCT::~FastDCT() {}

void codec::FastDCT::dct_inplace(nn::Tensor<int16_t, 3> input) const {
  const int channels = input.dimension(0);
  const int rows = input.dimension(1);
  const int cols = input.dimension(2);

  int16_t *data = work_buffer_.get();

  for (int channel = 0; channel < channels; channel++) {
    for (int row_offset = 0; row_offset < rows; row_offset += 8) {
      for (int col_offset = 0; col_offset < cols; col_offset += 8) {
        // fill buffer
        for (int row = 0; row < 8; row++) {
          for (int col = 0; col < 8; col++) {
            data[8 * row + col] =
                input(channel, row_offset + row, col_offset + col) - 128;
          }
        }

        // perform dct and scale
        // jsimd_fdct_ifast_sse2(data);
        jsimd_fdct_islow_sse2(data);
        jsimd_quantize_sse2(data, divisors_.get(), data);

        // put data back into tensor
        for (int row = 0; row < 8; row++) {
          for (int col = 0; col < 8; col++) {
            input(channel, row_offset + row, col_offset + col) =
                data[8 * row + col];
          }
        }
      }
    }
  }
}

// expects inputs between [0, 255]
nn::Tensor<int16_t, 3> codec::FastDCT::operator()(
    const nn::Tensor<int16_t, 3> input) const {
  nn::Tensor<int16_t, 3> output(input.deepcopy());
  dct_inplace(output);

  return std::move(output);
}

codec::FastIDCT::FastIDCT()
    : dct_table_(std::move(tj_data(sizeof(dct_table), dct_table))),
      work_buffer_(static_cast<int16_t *>(std::aligned_alloc(alignment, 128)),
                   [](void *ptr) { std::free(ptr); }) {}

codec::FastIDCT::~FastIDCT() {}

nn::Tensor<uint8_t, 3> codec::FastIDCT::operator()(
    const nn::Tensor<int16_t, 3> input) const {
  const int channels = input.dimension(0);
  const int rows = input.dimension(1);
  const int cols = input.dimension(2);

  nn::Tensor<uint8_t, 3> output(channels, rows, cols);

  int16_t *data = work_buffer_.get();
  uint8_t *outdata[8];

  for (int channel = 0; channel < channels; channel++) {
    for (int row_offset = 0; row_offset < rows; row_offset += 8) {
      for (int col_offset = 0; col_offset < cols; col_offset += 8) {
        // fill buffer
        for (int row = 0; row < 8; row++) {
          for (int col = 0; col < 8; col++) {
            data[8 * row + col] =
                input(channel, row_offset + row, col_offset + col);
          }
        }

        outdata[0] = &output(channel, row_offset, col_offset);
        outdata[1] = &output(channel, row_offset + 1, col_offset);
        outdata[2] = &output(channel, row_offset + 2, col_offset);
        outdata[3] = &output(channel, row_offset + 3, col_offset);
        outdata[4] = &output(channel, row_offset + 4, col_offset);
        outdata[5] = &output(channel, row_offset + 5, col_offset);
        outdata[6] = &output(channel, row_offset + 6, col_offset);
        outdata[7] = &output(channel, row_offset + 7, col_offset);

        // perform idct and descale
        // jsimd_idct_ifast_sse2(dct_table_.get(), data, outdata, 0);
        jsimd_idct_islow_sse2(dct_table_.get(), data, outdata, 0);
      }
    }
  }

  return std::move(output);
}
