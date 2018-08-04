#include <iostream>
#include <memory>
#include <vector>

#include "arithmetic_encoder.hh"

class InfiniteBitVector {
private:
    std::vector<char> bitvector_;
    size_t number_of_bits_;
    
public:
    InfiniteBitVector() :
        bitvector_(),
        number_of_bits_(0)
    { }

    InfiniteBitVector(std::vector<char> bitvector, size_t num_bits) :
        bitvector_(bitvector),
        number_of_bits_(num_bits)
    {
        if (num_bits / 8 >= bitvector.size()) {
            throw std::runtime_error("bitvector size not consistent with num_bits");
        }
    }

    ~InfiniteBitVector() { }
    
    void push_back_bit(uint8_t bit) {

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

    uint8_t get_bit(size_t bit_idx) {

        const int bit_offset = bit_idx % 8;
        const size_t byte_offset = bit_idx / 8;

        if (byte_offset >= bitvector_.size()) {
            throw std::runtime_error("bit index out of range");
        }
        
        return (bitvector_[byte_offset] & (1 << bit_offset)) >> bit_offset;
    }
    
    size_t size() {
        return number_of_bits_;
    }
    
    std::vector<char> vector() {
        return bitvector_;
    }
    
};

std::vector<char> codec::arith_encode(std::vector<char> input) {
  std::vector<char> output;

  // const std::vector<uint64_t> numerator = {23, 7, 1}; // A, B, EOM

  const uint64_t denominator = 23 + 7 + 1;
  const std::vector<std::pair<uint64_t, uint64_t>> numerator = {
      {0, 7},
      {7, 30},
      {30, 31}
  };

  std::cout << denominator << std::endl; 
  
  InfiniteBitVector bitvector;
  
  uint32_t high = 0xFFFFFFFF;
  uint32_t low = 0;
  input.push_back('$'); // add end of message symbol
  for (size_t i = 0; i < input.size(); i++) {

      uint8_t sym = 0;
      switch (input[i]) {
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
      
      // encode
      std::cout << static_cast<int>(sym) << std::endl;

      uint64_t range = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;
      std::cout << range << std::endl;

      // ...
      
  }
  
  return bitvector.vector();
}

std::vector<char> codec::arith_decode(std::vector<char> input) {
  std::vector<char> output;
  for (size_t i = 0; i < input.size(); i++) {
    output.push_back(input[i]);
  }

  return output;
}
