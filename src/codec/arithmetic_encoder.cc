#include <bitset>
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

#include "arithmetic_encoder.hh"

// A help class that gives you a std::vector like interface but for
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
    {0, 13},
    {13, 31},
    {31, 33},
};
const static uint64_t denominator = numerator[numerator.size() - 1].second;

// some nice constants
const int num_working_bits = 32;
static_assert(num_working_bits <= 63);
static_assert(num_working_bits >= 32);

const uint64_t max_range = static_cast<uint64_t>(1) << num_working_bits;
const uint64_t min_range = (max_range >> 2) + 2;
const uint64_t max = max_range - 1; // 0xFFFFFFFF for 32 working bits
const uint64_t min = 0;

const uint64_t top_mask = static_cast<uint64_t>(1) << (num_working_bits - 1);
const uint64_t second_mask = static_cast<uint64_t>(1) << (num_working_bits - 2);
const uint64_t mask = max; 

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
        
        const uint64_t new_high = low + (sym_high * range) / denominator - 1;
        const uint64_t new_low = low + (sym_low * range) / denominator;

        // if(sym != 2){
        //     std::cout << "next " << low + (numerator[sym+1].first * range) / denominator - 1 << std::endl;
        //     std::cout << "curr " << new_high << std::endl;
        // }
        
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
    std::cout << "pending_bits " << pending_bits << std::endl;    
    // bitvector.print();
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
    // std::cout << "value " << std::bitset<64>(value) << std::endl;
    // std::cout << "value                                 " << std::bitset<32>(value) << std::endl;
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




// std::vector<char> codec::arith_encode(const std::vector<char> input_) {

//     std::vector<char> input(input_);
//     input.push_back('$'); // add end of message symbol
    
//     InfiniteBitVector bitvector;
  
//    uint32_t high = 0xFFFFFFFFU;
//    uint32_t low = 0;

//    uint64_t pending_bits = 0;
//    for (size_t i = 0; i < input.size(); i++) {
//        std::cout << "iter " << i << std::endl;

//        const char character = input[i];
//        std::cout << "encoding " << character << std::endl;

//        uint8_t sym = 0;
//        switch (character) {
//        case 'A':
//            sym = 0;
//            break;
//        case 'B':
//            sym = 1;
//            break;
//        case '$':
//            sym = 2;
//            break;
//        default:
//            throw std::runtime_error("unrecognized symbol");
//        }

//        const uint64_t range = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;
//        std::cout << "  old high " << static_cast<double>(high) / 0xFFFFFFFFU << " " << std::bitset<32>(high) << std::endl;
//        std::cout << "  old low " << static_cast<double>(low) / 0xFFFFFFFFU << " " << std::bitset<32>(low) << std::endl;
//        std::cout << "  range " << std::bitset<33>(range) << std::endl;

//        high = low + (range * numerator[sym].second) / denominator - 1;
//        low = low + (range * numerator[sym].first) / denominator;
//        assert(high > low);
//        std::cout << "  high " << static_cast<double>(high) / 0xFFFFFFFFU << " " << std::bitset<32>(high) << std::endl;
//        std::cout << "  low " << static_cast<double>(low) / 0xFFFFFFFFU << " " << std::bitset<32>(low) << std::endl;

//        while (true) {
//            assert(high > low);

//            if (high < 0x80000000U) {
//                std::cout << "  emitting1 " << 0 << std::endl;
//                bitvector.push_back_bit(0);
//                for(size_t pending = 0; pending < pending_bits; pending++) {
//                    std::cout << "  emitting1 " << 1 << std::endl;
//                    bitvector.push_back_bit(1);
//                }
//                pending_bits = 0;              

//                std::cout << "  high " << static_cast<double>(high) / 0xFFFFFFFFU << " " << std::bitset<32>(high) << std::endl;
//                std::cout << "  low " << static_cast<double>(low) / 0xFFFFFFFFU << " " << std::bitset<32>(low) << std::endl;

//                // shift in a '0'
//                low <<= 1; 

//                // shift in a '1'
//                high <<= 1; 
//                high |= 1;

//            }
//            else if (low >= 0x80000000U) {
//                std::cout << "  emitting2 " << 1 << std::endl;              
//                bitvector.push_back_bit(1);
//                for(size_t pending = 0; pending < pending_bits; pending++) {
//                    std::cout << "  emitting2 " << 0 << std::endl;
//                    bitvector.push_back_bit(0);
//                }
//                pending_bits = 0;              

//                std::cout << "  high " << static_cast<double>(high) / 0xFFFFFFFFU << " " << std::bitset<32>(high) << std::endl;
//                std::cout << "  low " << static_cast<double>(low) / 0xFFFFFFFFU << " " << std::bitset<32>(low) << std::endl;
               
//                // shift in a '0'
//                low <<= 1;

//                // shift in a '1'
//                high <<= 1;
//                high |= 1;


//            }
//            else if (low >= 0x40000000U && high < 0xC0000000U) {
//                std::cout << "  expanding interval " << std::endl;
//                pending_bits++;

//                low <<= 1;
//                low &= 0x7FFFFFFFU;

//                high <<= 1;
//                high |= 0x80000001U;
               
//                // std::cout << "  high " << static_cast<double>(high) / 0xFFFFFFFFU << " " << std::bitset<32>(high) << std::endl;
//                // std::cout << "  low " << static_cast<double>(low) / 0xFFFFFFFFU << " " << std::bitset<32>(low) << std::endl;
//            }
//            else {
//                break;
//            }
//        }
//    }

//    std::cout << "pending bits " << pending_bits  << std::endl;

//    // output the final bits needed to encode the final symbols
//    pending_bits++;
//    if (low < 0x40000000U) {
//        std::cout << "  emitting " << 0 << std::endl;
//        bitvector.push_back_bit(0);
//        for(size_t pending = 0; pending < pending_bits; pending++) {
//            std::cout << "  emitting " << 1 << std::endl;
//            bitvector.push_back_bit(1);
//        }
//        pending_bits = 0;              
//    }
//    else {
//        std::cout << "  emitting " << 1 << std::endl;
//        bitvector.push_back_bit(1);
//        for(size_t pending = 0; pending < pending_bits; pending++) {
//            std::cout << "  emitting " << 0 << std::endl;
//            bitvector.push_back_bit(0);
//        }
//        pending_bits = 0;              
//    }

//    std::cout << "compressed size (bits): " << bitvector.size() << std::endl;
//    bitvector.print();

//    return bitvector.vector();
// }

// std::vector<char> codec::arith_decode(const std::vector<char> input, const size_t) {

//     const InfiniteBitVector bitvector(input);
//     std::vector<char> output;

//     uint32_t high = 0xFFFFFFFFU;
//     uint32_t low = 0;
//     uint64_t value = 0;
//     size_t bit_idx = 0; 

//     // pack the first value
//     for(size_t i = 0; i < 32 and i < bitvector.size(); i++) {
//         value |= (0xFFFFFFFFU & (bitvector.get_bit(i) << (31 - i)));
//         bit_idx++;
//     }

//     // while(true) {
//     for(size_t jj = 0; jj < 2048; jj++) {
//         std::cout << "iter " << jj << std::endl;

//         std::cout << "  high " << static_cast<double>(high) / 0xFFFFFFFFU << " " << std::bitset<32>(high) << std::endl;
//         std::cout << "  low " << static_cast<double>(low) / 0xFFFFFFFFU << " " << std::bitset<32>(low) << std::endl;
//         std::cout << "  value " << static_cast<double>(value) / 0xFFFFFFFFU << " " << std::bitset<64>(value) << std::endl;
//         assert(value >= low);
//         assert(value < high);        
//         const uint64_t range = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;
//         std::cout << "range " << static_cast<double>(range) / 0xFFFFFFFFU << std::endl;
        
//         int sym;
//         for(sym = 0; sym < 3; sym++) {
//             uint64_t sym_high = low + numerator[sym].second * range / denominator - 1;
//             uint64_t sym_low = low + numerator[sym].first * range / denominator;
//             assert(sym_high > sym_low);
            
//             std::cout << "  sym_high " << static_cast<double>(sym_high) / 0xFFFFFFFFU << " " << std::bitset<32>(sym_high) << std::endl;
//             std::cout << "  sym_low " << static_cast<double>(sym_low) / 0xFFFFFFFFU << " " << std::bitset<32>(sym_low) << std::endl;
//             std::cout << "  value " << static_cast<double>(value) / 0xFFFFFFFFU << " " << std::bitset<32>(value) << std::endl;
//             if (value >= sym_low and value < sym_high) {
//                 std::cout << "  break sym " << sym << std::endl;
//                 high = sym_high;
//                 low = sym_low;
//                 break;
//             }
//         }
//         assert(high > low);
//         assert(value >= low);
//         assert(value < high);
//         assert(sym == 0 or sym == 1 or sym == 2);
            
//         if (sym == 2) {
//             std::cout << "$ detected! done decoding!" << std::endl;
//             break;
//         }

//         switch(sym) {
//         case 0:
//             output.push_back('A');
//             // std::cout << "A detected!" << std::endl;            
//             break;
//         case 1:
//             output.push_back('B');
//             // std::cout << "B detected!" << std::endl;            
//             break;
//         default:
//             throw std::runtime_error("unrecognized symbol " + std::to_string(sym));
//         }       

//         std::cout << "  high " << static_cast<double>(high) / 0xFFFFFFFFU << " " << std::bitset<32>(high) << std::endl;
//         std::cout << "  low " << static_cast<double>(low) / 0xFFFFFFFFU << " " << std::bitset<32>(low) << std::endl;
//         std::cout << "  value " << static_cast<double>(value) / 0xFFFFFFFFU << " " << std::bitset<32>(value) << std::endl;

//         while (true) {
//             if (low >= 0x80000000U or high < 0x80000000U) {
//                 std::cout << "  consume bit1 " << std::endl;
//                 low <<= 1;
//                 high <<= 1;
//                 high |= 1;

//                 value <<= 1;
//                 if (bit_idx < bitvector.size() or true) {
//                     uint32_t bit = bitvector.get_bit(bit_idx);
//                     bit_idx++;
                    
//                     std::cout << "  pop bit1 " << bit << std::endl;
//                     value += bit;
//                 }
//                 value &= 0xFFFFFFFFU;                
                
//                 std::cout << "  high " << static_cast<double>(high) / 0xFFFFFFFFU << " " << std::bitset<32>(high) << std::endl;
//                 std::cout << "  low " << static_cast<double>(low) / 0xFFFFFFFFU << " " << std::bitset<32>(low) << std::endl;
//                 std::cout << "  value " << static_cast<double>(value) / 0xFFFFFFFFU << " " << std::bitset<32>(value) << std::endl;
//             }
//             else if (low >= 0x40000000U && high < 0xC0000000U) {
//                 std::cout << "  consume bit2 " << std::endl;

//                 low <<= 1;
//                 low &= 0x7FFFFFFF;
//                 high <<= 1;
//                 high |= 0x80000001;

//                 value <<= 1;
//                 if (bit_idx < bitvector.size() or true) {
//                     uint32_t bit = bitvector.get_bit(bit_idx);
//                     bit_idx++;
                    
//                     std::cout << "  pop bit1 " << bit << std::endl;
//                     value += bit;
//                 }
//                 value &= 0xFFFFFFFFU;

//                 std::cout << "  high " << static_cast<double>(high) / 0xFFFFFFFFU << " " << std::bitset<32>(high) << std::endl;
//                 std::cout << "  low " << static_cast<double>(low) / 0xFFFFFFFFU << " " << std::bitset<32>(low) << std::endl;
//                 std::cout << "  value " << static_cast<double>(value) / 0xFFFFFFFFU << " " << std::bitset<32>(value) << std::endl;
//             }
//             else {
//                 break;
//             }
            
//         }
//         std::cout << "  exited loop" << std::endl;

//     }
    
//     return output;
// }
