#ifndef _CODEC_ARITHMETIC_ENCODER_HH
#define _CODEC_ARITHMETIC_ENCODER_HH

#include <cassert>
#include <memory>
#include <vector>

namespace codec {

    const static std::vector<std::pair<uint32_t, uint32_t>> numerator = {
        {0, 11000},
        {11000, 11999},
        {11999, 12000},
    };
    const static uint32_t denominator = numerator[numerator.size() - 1].second;

    // A helper class that gives you a std::vector like interface but
    // for individual bits.
    class InfiniteBitVector {
    private:
        std::vector<char> bitvector_;
        size_t number_of_bits_;

    public:
        InfiniteBitVector() :
            bitvector_(),
            number_of_bits_(0)
        { }

        InfiniteBitVector(std::vector<char> bitvector) :
            bitvector_(bitvector),
            number_of_bits_(8*bitvector.size())
        { }

        ~InfiniteBitVector() { }

        void push_back_bit(const uint8_t bit) {

            if (bit != 0 and bit != 1) {
                throw std::runtime_error("bit must be 0 or 1");
            }

            const int bit_offset = number_of_bits_ % 8;
            if (bit_offset == 0) {
                bitvector_.push_back(0);
            }
            const size_t byte_offset = bitvector_.size() - 1;

            bitvector_[byte_offset] |= (bit << bit_offset);

            ++number_of_bits_;
        }

        uint8_t get_bit(const size_t bit_idx) const {

            const int bit_offset = bit_idx % 8;
            const size_t byte_offset = bit_idx / 8;

            if (byte_offset >= bitvector_.size()) {
                throw std::runtime_error("bit index out of range");
            }

            return (bitvector_[byte_offset] & (1 << bit_offset)) >> bit_offset;
        }

        size_t size() const {
            return number_of_bits_;
        }

        std::vector<char> vector() const {
            return bitvector_;
        }

        void print() const {
            for (size_t i = 0; i < this->size(); i++) {
                std::cout << static_cast<int>(this->get_bit(i));
            }
            std::cout << std::endl;        
        }
    };

    class Encoder_ {
    private:
        InfiniteBitVector bitvector_;
        uint64_t pending_bits_;
        
    public:
        Encoder_() :
            bitvector_(),
            pending_bits_(0)
        { }
        
        ~Encoder_() { }

        void shift(uint8_t bit) {

            bitvector_.push_back_bit(bit);
            
            // the pending bits will be the opposite of the shifted
            // bit.
            for (; pending_bits_ > 0; pending_bits_--) {
                bitvector_.push_back_bit(bit ^ 0x1);
            }
            assert(pending_bits_ == 0);
                
        }

        void underflow() {
            assert(pending_bits_ < std::numeric_limits<decltype(pending_bits_)>::max());
            pending_bits_ += 1;
        }

        InfiniteBitVector bitvector() const {
            return bitvector_;
        }
    };

    class Decoder_ {
    private:
        
    public:
        Decoder_() {}
        ~Decoder_() {}

        void shift(uint64_t) {
            
        }
        void underflow() {

        }
        
    };
    
    template<class T>
    class Coder_ {
    private:
        static constexpr uint64_t num_working_bits_ = 31;
        static_assert(num_working_bits_ < 63);
        
        static constexpr uint64_t max_range_ = static_cast<uint64_t>(1) << num_working_bits_;
        static constexpr uint64_t min_range_ = (max_range_ >> 2) + 2;
        static constexpr uint64_t max_ = max_range_ - 1; // 0xFFFFFFFF for 32 working bits
        static constexpr uint64_t min_ = 0;
        
        static constexpr uint64_t top_mask_ = static_cast<uint64_t>(1) << (num_working_bits_ - 1);
        static constexpr uint64_t second_mask_ = static_cast<uint64_t>(1) << (num_working_bits_ - 2);
        static constexpr uint64_t mask_ = max_; 

        uint64_t high_;
        uint64_t low_;
        uint64_t range_;
        
        T coder_;

    protected:
        Coder_() :
            high_(max_),
            low_(min_),
            range_(min_),
            coder_()
        { }

        ~Coder_() { }
        
        const T& get_coder() const {
            return coder_;
        }

        inline void advance(const uint32_t sym_low, const uint32_t sym_high, const uint32_t denominator) {

            const uint64_t range = high_ - low_ + 1;
            assert(range <= max_range_);
            assert(range >= min_range_);

            // check if overflow would happen
            assert((range == 0) || (high_ < (std::numeric_limits<uint64_t>::max() / range)));
            assert((range == 0) || (low_ < (std::numeric_limits<uint64_t>::max() / range)));

            const uint64_t new_high = low_ + (sym_high * range) / denominator - 1;
            const uint64_t new_low = low_ + (sym_low * range) / denominator;
            
            assert(new_high <= max_);
            assert(new_low <= max_);
            assert(new_high == (mask_ & new_high));
            assert(new_low == (mask_ & new_low));
            
            high_ = new_high;
            low_ = new_low;
            assert(high_ > low_);        
            
            while(true) {
                // if the MSB of both numbers match, then shift out a bit
                // into the vector and shift out all `pending bits`.
                if (((high_ ^ low_) & top_mask_) == 0) {

                    // grab the MSB of `low` (will be the same as `high`)
                    const uint8_t bit = (low_ >> (num_working_bits_ - 1));
                    assert(bit <= 0x1);
                    assert(bit == (high_ >> (num_working_bits_ - 1)));
                    assert(static_cast<uint8_t>(!!(high_ & top_mask_)) == bit);

                    coder_.shift(bit);

                    low_ = (low_ << 1) & mask_;
                    high_ = ((high_ << 1) & mask_) | 0x1;

                    assert(high_ <= max_);
                    assert(low_ <= max_);
                    assert(high_ > low_);
                }
                // the second highest bit of `high` is a 1 and the second
                // highest bit of `low` is 0, then the `low` and `high`
                // are converging. To handle this, we increment
                // `pending_bits` and shift `high` and `low`. The value
                // true value of the shifted bits will be determined once
                // the MSB bits match after consuming more symbols. 
                else if ((low_ & ~high_ & second_mask_) != 0) {

                    coder_.underflow();

                    low_ = (low_ << 1) & (mask_ >> 1); 
                    high_ = ((high_ << 1) & (mask_ >> 1)) | top_mask_ | 0x1;

                    assert(high_ <= max_);
                    assert(low_ <= max_);
                    assert(high_ > low_);
                }
                else {
                    break;
                }
            }

        }

    };

    class ArithmeticEncoder : Coder_<Encoder_> {
    private:
        const Encoder_& coder_;
        bool finished;
        
    public:
        ArithmeticEncoder() :
            coder_(get_coder()),
            finished(false)
        { }

        ~ArithmeticEncoder() {}

        std::vector<char> encode (std::vector<char> input) {
            
            for (auto symbol : input) {
                const uint32_t sym_low = numerator[symbol].first;
                const uint32_t sym_high = numerator[symbol].second;
                const uint32_t denom = denominator;
                
                advance(sym_low, sym_high, denom);
            }

            InfiniteBitVector encoded_data = coder_.bitvector();
            encoded_data.push_back_bit(0x1);
            
            return encoded_data.vector();
        }
    };

    
    class ArithmeticDecoder : Coder_<Decoder_> {
    private:
        const Decoder_& coder_;

        std::vector<char> decode (std::vector<char> input) {

            InfiniteBitVector encoded_data(input);
            
            

        }
        
        
    public:

    };

    std::vector<char> arith_encode(const std::vector<char> input);
    std::vector<char> arith_decode(const std::vector<char> input);
}

#endif  // _CODEC_ARITHMETIC_ENCODER_HH
