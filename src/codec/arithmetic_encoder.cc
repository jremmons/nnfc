#include <bitset>
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

#include "arithmetic_encoder.hh"

// A helper class that gives you a std::vector like interface but for
// individual bits. 
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

// my current implementation uses this hard-coded probability model
const static std::vector<std::pair<uint64_t, uint64_t>> numerator = {
    {0, 11000},
    {11000, 11999},
    {11999, 12000},
};
const static uint64_t denominator = numerator[numerator.size() - 1].second;

// some nice constants
const int num_working_bits = 32;
const uint64_t max_range = static_cast<uint64_t>(1) << num_working_bits;
const uint64_t min_range = (max_range >> 2) + 2;
const uint64_t max = max_range - 1; // 0xFFFFFFFF for 32 working bits
const uint64_t min = 0;

const uint64_t top_mask = static_cast<uint64_t>(1) << (num_working_bits - 1);
const uint64_t second_mask = static_cast<uint64_t>(1) << (num_working_bits - 2);
const uint64_t mask = max; 

// static_assert(/* max deminator */ < (std::numeric_limits<uint64_t>::max() / max_range));

// the encoder
std::vector<char> codec::arith_encode(const std::vector<char> input_) {
    std::vector<char> input(input_);
    input.push_back('$');
    
    InfiniteBitVector bitvector;    
    uint64_t high = max;
    uint64_t low = min;
    uint64_t pending_bits = 0;
    
    for (size_t i = 0; i < input.size(); i++) {
        const char character = input[i];
        
        uint32_t sym = 0;
        switch (character) {
        case 'A':
            sym = 0;
            break;
        case 'B':
            sym = 1;
           break;
        case '$':
            sym = 2;
            break;
        default:
            throw std::runtime_error("unrecognized symbol");
        }
        assert(sym == 0 or sym == 1 or sym == 2);

        const uint64_t range = high - low + 1;
        assert(range <= max_range);
        assert(range >= min_range);

        const uint64_t sym_high = numerator[sym].second;
        const uint64_t sym_low = numerator[sym].first;
        assert(sym_high > sym_low);
        assert(sym_high < max);
        assert(sym_low < max);

        // check if overflow is possible
        assert((sym_high == 0) || (sym_high < (std::numeric_limits<uint64_t>::max() / range)));
        assert((sym_low == 0) || (sym_low < (std::numeric_limits<uint64_t>::max() / range)));
        const uint64_t new_high = low + (sym_high * range) / denominator - 1;
        const uint64_t new_low = low + (sym_low * range) / denominator;

        assert(new_high <= max);
        assert(new_low <= max);
        assert(new_high == (mask & new_high));
        assert(new_low == (mask & new_low));
        
        high = new_high;
        low = new_low;
        assert(high > low);        

        while(true) {

            // if the MSB of both numbers match, then shift out a bit
            // into the vector and shift out all `pending bits`.
            if (((high ^ low) & top_mask) == 0) {

                // grab the MSB of `low` (will be the same as `high`)
                const uint8_t bit = (low >> (num_working_bits - 1));
                assert(bit <= 0x1);
                assert(bit == (high >> (num_working_bits - 1)));
                assert(!!(high & top_mask) == bit);
                
                bitvector.push_back_bit(bit);

                // the pending bits will be the opposite of the
                // shifted bit.
                for (; pending_bits > 0; pending_bits--) {
                    bitvector.push_back_bit(bit ^ 0x1);
                }
                assert(pending_bits == 0);
                
                low = (low << 1) & mask;
                high = ((high << 1) & mask) | 0x1;

                assert(high <= max);
                assert(low <= max);
                assert(high > low);
            }
            // the second highest bit of `high` is a 1 and the second
            // highest bit of `low` is 0, then the `low` and `high`
            // are converging. To handle this, we increment
            // `pending_bits` and shift `high` and `low`. The value
            // true value of the shifted bits will be determined once
            // the MSB bits match after consuming more symbols. 
            else if ((low & ~high & second_mask) != 0) {
                
                assert(pending_bits < std::numeric_limits<decltype(pending_bits)>::max());
                pending_bits += 1;

                low = (low << 1) & (mask >> 1); 
                high = ((high << 1) & (mask >> 1)) | top_mask | 1;

                assert(high <= max);
                assert(low <= max);
                assert(high > low);
            }
            else {
                break;
            }
        }                
    }

    // finalize the bitvector but add `epsilon` to the end.
    bitvector.push_back_bit(0x1);

    return bitvector.vector();
}


std::vector<char> codec::arith_decode(const std::vector<char> input_) {
    std::vector<char> output;

    char syms[] = {'A', 'B', '$'};
    
    size_t bit_idx = 0;
    const InfiniteBitVector bitvector(input_);

    uint64_t high = max;
    uint64_t low = min;    
    uint64_t value = 0;
    
    for(size_t i = 0; i < num_working_bits and i < bitvector.size(); i++) {
        value |= (static_cast<uint64_t>(bitvector.get_bit(i)) << (num_working_bits - i - 1));
        bit_idx++;
    }
    assert(value <= max);

    while(true) {
        const uint64_t range = high - low + 1;
        assert(range <= max_range);
        assert(range >= min_range);

        int sym = -1;
        for(uint64_t sym_idx = 0; sym_idx < 3; sym_idx++) {

            const uint64_t sym_high = low + (numerator[sym_idx].second * range) / denominator - 1; 
            const uint64_t sym_low = low + (numerator[sym_idx].first * range) / denominator;

            // std::cout << "sym_idx " << sym_idx << std::endl; 
            // std::cout << "high  " << std::bitset<64>(sym_high) << std::endl;
            // std::cout << "value " << std::bitset<64>(value) << std::endl;
            // std::cout << "low   " << std::bitset<64>(sym_low) << std::endl;
            
            if (value < sym_high and value >= sym_low) {
                sym = sym_idx;
                high = sym_high;
                low = sym_low;
                // std::cout << "break" << std::endl;
                break;
            }
        }

        if (sym == -1) {
            throw std::runtime_error("decoding error");
        }
        if (sym == 2) {
            std::cout << "$ decoded! end!" << std::endl; 
            break;
        }

        // std::cout << "decoded " << syms[sym] << std::endl;
        output.push_back(syms[sym]);

        while(true) {

            // if the MSB of both numbers match, then shift out a bit
            // into the vector and shift out all `pending bits`.
            if (((high ^ low) & top_mask) == 0) {

                // grab the MSB of `low` (will be the same as `high`)
                const uint8_t bit = (low >> (num_working_bits - 1));
                assert(bit <= 0x1);
                assert(bit == (high >> (num_working_bits - 1)));
                assert(!!(high & top_mask) == bit);

                assert(((value & top_mask) >> (num_working_bits - 1)) == bit);
                
                uint8_t bitvector_bit = 0;
                if (bit_idx < bitvector.size()) {
                    bitvector_bit = static_cast<uint64_t>(bitvector.get_bit(bit_idx)) & 0x1;
                }
                value = ((value << 1) & mask) | bitvector_bit;
                assert(value <= max);
                bit_idx++;
                
                low = (low << 1) & mask;
                high = ((high << 1) & mask) | 0x1;

                assert(high <= max);
                assert(low <= max);
                assert(high > low);
            }
            // the second highest bit of `high` is a 1 and the second
            // highest bit of `low` is 0, then the `low` and `high`
            // are converging. To handle this, we increment
            // `pending_bits` and shift `high` and `low`. The value
            // true value of the shifted bits will be determined once
            // the MSB bits match after consuming more symbols. 
            else if ((low & ~high & second_mask) != 0) {
                
                uint64_t bitvector_bit = 0;
                if (bit_idx < bitvector.size()) {
                    bitvector_bit = static_cast<uint64_t>(bitvector.get_bit(bit_idx)) & 0x1;
                }
                value = (value & top_mask) | ((value << 1) & (mask >> 1)) | bitvector_bit;
                bit_idx++;
                assert(value <= max);

                low = (low << 1) & (mask >> 1); 
                high = ((high << 1) & (mask >> 1)) | top_mask | 1;

                assert(high <= max);
                assert(low <= max);
                assert(high > low);
            }
            else {
                break;
            }
        }                
    }

    return output;
}
