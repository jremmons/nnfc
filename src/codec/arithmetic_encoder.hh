#ifndef _CODEC_ARITHMETIC_ENCODER_HH
#define _CODEC_ARITHMETIC_ENCODER_HH

#include <memory>
#include <vector>

namespace codec {

std::vector<char> arith_encode(const std::vector<char> input);
    std::vector<char> arith_decode(const std::vector<char> input, const size_t num_bits);
}

#endif  // _CODEC_ARITHMETIC_ENCODER_HH
