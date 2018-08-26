#include <cstdlib>
#include <memory>

#include "nn/tensor.hh"
#include "fastdct.hh"
#include "tjdct/jsimd.hh"

inline std::unique_ptr<int16_t> tj_data(const int size_, const int16_t* data_) {
    const int alignment = 64; // use 64 byte alignment in case we ever use AVX512 acceleration

    std::unique_ptr<int16_t> data(static_cast<int16_t*>(std::aligned_alloc(alignment, size_)));
    std::memcpy(data.get(), data_, size_);
    
    return std::move(data);
}

codec::FastDCT::FastDCT() :
    divisors_(std::move(tj_data(sizeof(divisors), divisors)))
{}

codec::FastDCT::~FastDCT() {}

void codec::FastDCT::dct_inplace(nn::Tensor<int16_t, 3>) const {
    // apply the dct and quantization to each element in each channel.
}

nn::Tensor<int16_t, 3> codec::FastDCT::dct(const nn::Tensor<int16_t, 3> input) const {
    nn::Tensor<int16_t, 3> output(input);
    dct_inplace(output);

    return std::move(output);
}

codec::FastIDCT::FastIDCT() :
    dct_table_(std::move(tj_data(sizeof(dct_table), dct_table)))
{}

codec::FastIDCT::~FastIDCT() {}

void codec::FastIDCT::idct_inplace(nn::Tensor<int16_t, 3>) const {
    // apply the idct to each element in each channel.
}

nn::Tensor<int16_t, 3> codec::FastIDCT::idct(const nn::Tensor<int16_t, 3> input) const {
    nn::Tensor<int16_t, 3> output(input);
    idct_inplace(output);

    return std::move(output);
}
