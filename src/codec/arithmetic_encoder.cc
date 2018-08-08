#include <bitset>
#include <cassert>
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
            return this->get_bit(bitvector_.size() - 1);
            // throw std::runtime_error("bit index out of range");
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
            //std::cout << i << " " << static_cast<int>(this->get_bit(i)) << "\n";
            std::cout << static_cast<int>(this->get_bit(i));
        }
        std::cout << std::endl;        
    }
    
};

const static std::vector<std::pair<uint64_t, uint64_t>> numerator = {
    {0, 10},
    {10, 11},
    {11, 12}
};
const static uint64_t denominator = numerator[numerator.size() - 1].second;

std::vector<char> codec::arith_encode(const std::vector<char> input_) {

    std::vector<char> input(input_);
    input.push_back('$'); // add end of message symbol
    
    InfiniteBitVector bitvector;
  
   uint32_t high = 0xFFFFFFFFU;
   uint32_t low = 0;

   uint64_t pending_bits = 0;
   for (size_t i = 0; i < input.size(); i++) {
       std::cout << "iter " << i << std::endl;

       const char character = input[i];
       std::cout << "encoding " << character << std::endl;

       uint8_t sym = 0;
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

       const uint64_t range = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;
       std::cout << "  old high " << static_cast<double>(high) / 0xFFFFFFFFU << " " << std::bitset<32>(high) << std::endl;
       std::cout << "  old low " << static_cast<double>(low) / 0xFFFFFFFFU << " " << std::bitset<32>(low) << std::endl;
       std::cout << "  range " << std::bitset<33>(range) << std::endl;

       high = low + (range * numerator[sym].second) / denominator - 1;
       low = low + (range * numerator[sym].first) / denominator;
       assert(high > low);
       std::cout << "  high " << static_cast<double>(high) / 0xFFFFFFFFU << " " << std::bitset<32>(high) << std::endl;
       std::cout << "  low " << static_cast<double>(low) / 0xFFFFFFFFU << " " << std::bitset<32>(low) << std::endl;

       while (true) {
           assert(high > low);

           if (high < 0x80000000U) {
               std::cout << "  emitting1 " << 0 << std::endl;
               bitvector.push_back_bit(0);
               for(size_t pending = 0; pending < pending_bits; pending++) {
                   std::cout << "  emitting1 " << 1 << std::endl;
                   bitvector.push_back_bit(1);
               }
               pending_bits = 0;              

               std::cout << "  high " << static_cast<double>(high) / 0xFFFFFFFFU << " " << std::bitset<32>(high) << std::endl;
               std::cout << "  low " << static_cast<double>(low) / 0xFFFFFFFFU << " " << std::bitset<32>(low) << std::endl;

               // shift in a '0'
               low <<= 1; 

               // shift in a '1'
               high <<= 1; 
               high |= 1;

           }
           else if (low >= 0x80000000U) {
               std::cout << "  emitting2 " << 1 << std::endl;              
               bitvector.push_back_bit(1);
               for(size_t pending = 0; pending < pending_bits; pending++) {
                   std::cout << "  emitting2 " << 0 << std::endl;
                   bitvector.push_back_bit(0);
               }
               pending_bits = 0;              

               std::cout << "  high " << static_cast<double>(high) / 0xFFFFFFFFU << " " << std::bitset<32>(high) << std::endl;
               std::cout << "  low " << static_cast<double>(low) / 0xFFFFFFFFU << " " << std::bitset<32>(low) << std::endl;
               
               // shift in a '0'
               low <<= 1;

               // shift in a '1'
               high <<= 1;
               high |= 1;


           }
           else if (low >= 0x40000000U && high < 0xC0000000U) {
               std::cout << "  expanding interval " << std::endl;
               pending_bits++;

               low <<= 1;
               low &= 0x7FFFFFFFU;

               high <<= 1;
               high |= 0x80000001U;
               
               // std::cout << "  high " << static_cast<double>(high) / 0xFFFFFFFFU << " " << std::bitset<32>(high) << std::endl;
               // std::cout << "  low " << static_cast<double>(low) / 0xFFFFFFFFU << " " << std::bitset<32>(low) << std::endl;
           }
           else {
               break;
           }
       }
   }

   std::cout << "pending bits " << pending_bits  << std::endl;

   // output the final bits needed to encode the final symbols
   pending_bits++;
   if (low < 0x40000000U) {
       std::cout << "  emitting " << 0 << std::endl;
       bitvector.push_back_bit(0);
       for(size_t pending = 0; pending < pending_bits; pending++) {
           std::cout << "  emitting " << 1 << std::endl;
           bitvector.push_back_bit(1);
       }
       pending_bits = 0;              
   }
   else {
       std::cout << "  emitting " << 1 << std::endl;
       bitvector.push_back_bit(1);
       for(size_t pending = 0; pending < pending_bits; pending++) {
           std::cout << "  emitting " << 0 << std::endl;
           bitvector.push_back_bit(0);
       }
       pending_bits = 0;              
   }

   std::cout << "compressed size (bits): " << bitvector.size() << std::endl;
   bitvector.print();

   return bitvector.vector();
}

std::vector<char> codec::arith_decode(const std::vector<char> input, const size_t) {

    const InfiniteBitVector bitvector(input);
    std::vector<char> output;

    uint32_t high = 0xFFFFFFFFU;
    uint32_t low = 0;
    uint64_t value = 0;
    size_t bit_idx = 0; 

    // pack the first value
    for(size_t i = 0; i < 32 and i < bitvector.size(); i++) {
        value |= (0xFFFFFFFFU & (bitvector.get_bit(i) << (31 - i)));
        bit_idx++;
    }

    // while(true) {
    for(size_t jj = 0; jj < 2048; jj++) {
        std::cout << "iter " << jj << std::endl;

        std::cout << "  value " << static_cast<double>(value) / 0xFFFFFFFFU << " " << std::bitset<64>(value) << std::endl;
        assert(value >= low);
        assert(value < high);        
        const uint64_t range = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;
        std::cout << "range " << static_cast<double>(range) / 0xFFFFFFFFU << std::endl;
        
        int sym = 0;
        for(sym = 0; sym < 3; sym++) {
            uint64_t sym_low = numerator[sym].first * range / denominator + low;
            uint64_t sym_high = numerator[sym].second * range / denominator + low;

            std::cout << "  sym_high " << static_cast<double>(sym_high) / 0xFFFFFFFFU << " " << std::bitset<32>(sym_high) << std::endl;
            std::cout << "  sym_low " << static_cast<double>(sym_low) / 0xFFFFFFFFU << " " << std::bitset<32>(sym_low) << std::endl;
            std::cout << "  value " << static_cast<double>(value) / 0xFFFFFFFFU << " " << std::bitset<32>(value) << std::endl;
            if (value >= sym_low and value < sym_high) {
                std::cout << "  break sym " << sym << std::endl;
                break;
            }
        }
                
        if (sym == 2) {
            std::cout << "$ detected! done decoding!" << std::endl;
            break;
        }

        switch(sym) 
        {
        case 0:
            output.push_back('A');
            std::cout << "A detected!" << std::endl;            
            break;
        case 1:
            output.push_back('B');
            std::cout << "B detected!" << std::endl;            
            break;
        default:
            throw std::runtime_error("unrecognized symbol " + std::to_string(sym));
        }       
        
        high = low + (range * numerator[sym].second) / denominator - 1;
        low = low + (range * numerator[sym].first) / denominator;
        assert(high > low);
        std::cout << "  high " << static_cast<double>(high) / 0xFFFFFFFFU << " " << std::bitset<32>(high) << std::endl;
        std::cout << "  low " << static_cast<double>(low) / 0xFFFFFFFFU << " " << std::bitset<32>(low) << std::endl;
        std::cout << "  value " << static_cast<double>(value) / 0xFFFFFFFFU << " " << std::bitset<32>(value) << std::endl;

        while (true) {
            if (low >= 0x80000000U || high < 0x80000000U) {
                std::cout << "  consume bit1 " << std::endl;
                low <<= 1;
                high <<= 1;
                high |= 1;

                value <<= 1;
                if (bit_idx < bitvector.size()) {
                    uint32_t bit = bitvector.get_bit(bit_idx);
                    bit_idx++;
                    
                    std::cout << "  pop bit1 " << bit << std::endl;
                    value += bit;
                }
                value &= 0xFFFFFFFFU;                
                
                std::cout << "  high " << static_cast<double>(high) / 0xFFFFFFFFU << " " << std::bitset<32>(high) << std::endl;
                std::cout << "  low " << static_cast<double>(low) / 0xFFFFFFFFU << " " << std::bitset<32>(low) << std::endl;
                std::cout << "  value " << static_cast<double>(value) / 0xFFFFFFFFU << " " << std::bitset<32>(value) << std::endl;
            }
            else if (low >= 0x40000000U && high < 0xC0000000U) {
                std::cout << "  consume bit2 " << std::endl;

                low <<= 1;
                low &= 0x7FFFFFFF;
                high <<= 1;
                high |= 0x80000001;

                value <<= 1;
                if (bit_idx < bitvector.size()) {
                    uint32_t bit = bitvector.get_bit(bit_idx);
                    bit_idx++;
                    
                    std::cout << "  pop bit1 " << bit << std::endl;
                    value += bit;
                }
                value &= 0xFFFFFFFFU;                

                std::cout << "  high " << static_cast<double>(high) / 0xFFFFFFFFU << " " << std::bitset<32>(high) << std::endl;
                std::cout << "  low " << static_cast<double>(low) / 0xFFFFFFFFU << " " << std::bitset<32>(low) << std::endl;
                std::cout << "  value " << static_cast<double>(value) / 0xFFFFFFFFU << " " << std::bitset<32>(value) << std::endl;
            }
            else {
                break;
            }
            
        }
        std::cout << "  exited loop" << std::endl;

    }
    
    return output;
}
