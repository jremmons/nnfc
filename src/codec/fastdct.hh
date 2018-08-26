#ifndef _CODEC_FASTDCT_HH
#define _CODEC_FASTDCT_HH

#include <memory>
#include "nn/tensor.hh"

namespace codec {

class FastDCT {
private:
    const std::unique_ptr<int16_t> divisors_;
    
public:
    FastDCT();
    ~FastDCT();

    void dct_inplace(nn::Tensor<int16_t, 3> input) const;
    nn::Tensor<int16_t, 3> dct(const nn::Tensor<int16_t, 3> input) const;
};

class FastIDCT {
private:
    const std::unique_ptr<int16_t> dct_table_;

public:    
    FastIDCT();
    ~FastIDCT();

    nn::Tensor<int16_t, 3> idct(const nn::Tensor<int16_t, 3> input) const;
    void idct_inplace(nn::Tensor<int16_t, 3> input) const;
};

}

#endif // _CODEC_FASTDCT_HH
