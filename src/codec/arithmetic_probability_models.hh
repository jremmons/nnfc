#ifndef _CODEC_ARITHMETIC_PROBABILITY_MODELS_HH
#define _CODEC_ARITHMETIC_PROBABILITY_MODELS_HH

#include <cassert>
#include <memory>
#include <vector>

namespace codec {

class SimpleModel {
 private:
  const std::vector<std::pair<uint32_t, uint32_t>> numerator;
  const uint32_t denominator;

 public:
  SimpleModel()
      : numerator({
            {0, 11000}, {11000, 11999}, {11999, 12000},
        }),
        denominator(numerator[numerator.size() - 1].second) {}
  ~SimpleModel() {}

  inline std::pair<uint32_t, uint32_t> symbol_numerator(uint32_t symbol) const {
    assert(symbol <= 2);
    return numerator[symbol];
  }

  inline uint32_t symbol_denominator() const { return denominator; }

  inline uint32_t size() { return 3; }
};

}

#endif  // _CODEC_ARITHMETIC_PROBABILITY_MODELS_HH
