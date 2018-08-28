#ifndef _CODEC_ARITHMETIC_CODER_COMMON_HH
#define _CODEC_ARITHMETIC_CODER_COMMON_HH

// constants (trying to avoid polluting the `codec` namespace)
namespace codec {
namespace arithmetic_coder {
static constexpr uint64_t num_working_bits = 31;
static_assert(num_working_bits < 63);

static constexpr uint64_t max_range = static_cast<uint64_t>(1)
                                      << num_working_bits;
static constexpr uint64_t min_range = (max_range >> 2) + 2;
static constexpr uint64_t working_bits_max =
    max_range - 1;  // 0x7FFFFFFF for 31 working bits
static constexpr uint64_t working_bits_min = 0;

static constexpr uint64_t top_mask = static_cast<uint64_t>(1)
                                     << (num_working_bits - 1);
static constexpr uint64_t second_mask = static_cast<uint64_t>(1)
                                        << (num_working_bits - 2);
static constexpr uint64_t working_bits_mask = working_bits_max;
}
}

#endif  // _CODEC_ARITHMETIC_CODER_COMMON_HH

