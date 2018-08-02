#ifndef _CODEC_ARITHMETIC_ENCODER_HH
#define _CODEC_ARITHMETIC_ENCODER_HH

#include <memory>
#include <vector>

namespace codec {

    std::vector<char> arith_encode(std::vector<char> input);
    std::vector<char> arith_decode(std::vector<char> input);
    
}

#endif  // _CODEC_ARITHMETIC_ENCODER_HH
