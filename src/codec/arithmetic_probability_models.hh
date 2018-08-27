#ifndef _CODEC_ARITHMETIC_PROBABILITY_MODELS_HH
#define _CODEC_ARITHMETIC_PROBABILITY_MODELS_HH

#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include <json.hh>

namespace codec {

class SimpleModel {
 private:
  const std::vector<std::pair<uint32_t, uint32_t>> numerator_;
  const uint32_t denominator_;

 public:
  SimpleModel()
      : numerator_({
            {0, 11000}, {11000, 11999}, {11999, 12000},
        }),
        denominator_(numerator_[numerator_.size() - 1].second) {}
  ~SimpleModel() {}

  inline void consume_symbol(const uint32_t) {}

  inline std::pair<uint32_t, uint32_t> symbol_numerator(uint32_t symbol) const {
    assert(symbol <= 2);
    return numerator_[symbol];
  }

  inline uint32_t denominator() const { return denominator_; }

  inline uint32_t size() { return 3; }

  inline uint32_t finished_symbol() const { return 2; }
};

class SimpleAdaptiveModel {
 private:
  const uint32_t num_symbols_;
  std::vector<std::pair<uint32_t, uint32_t>> numerator_;
  uint32_t denominator_;

 public:
  SimpleAdaptiveModel(const uint32_t num_symbols)
      : num_symbols_(num_symbols + 1),
        numerator_(num_symbols + 1),
        denominator_(num_symbols + 1) {
    for (uint32_t i = 0; i < num_symbols_; i++) {
      numerator_[i].first = i;
      numerator_[i].second = i + 1;
    }
  }

  SimpleAdaptiveModel(const std::string probabilities_json)
      : num_symbols_(), numerator_(), denominator_() {
    nlohmann::json model_json = nlohmann::json::parse(probabilities_json);
  }

  ~SimpleAdaptiveModel() {}

  inline void consume_symbol(const uint32_t symbol) {
    denominator_ += 1;

    numerator_[symbol].second += 1;
    for (uint32_t i = symbol + 1; i < num_symbols_; i++) {
      const uint32_t range = numerator_[i].second - numerator_[i].first;

      numerator_[i].first = numerator_[i - 1].second;
      numerator_[i].second = numerator_[i].first + range;
    }

    assert(numerator_[num_symbols_ - 1].second == denominator_);
  }

  inline std::pair<uint32_t, uint32_t> symbol_numerator(
      const uint32_t symbol) const {
    assert(symbol <= num_symbols_);
    return numerator_[symbol];
  }

  inline uint32_t denominator() const { return denominator_; }

  inline uint32_t size() const { return num_symbols_; }

  inline uint32_t finished_symbol() const { return num_symbols_ - 1; }
};
}

#endif  // _CODEC_ARITHMETIC_PROBABILITY_MODELS_HH
