#include <iostream>
#include <memory>
#include <vector>

#include "arithmetic_encoder.hh"

std::vector<char> codec::arith_encode(std::vector<char> input){
    std::vector<char> output;
    for(size_t i = 0; i < input.size(); i++) {
        output.push_back(input[i]);
    }
    
    return output;
}

std::vector<char> codec::arith_decode(std::vector<char> input){
    std::vector<char> output;
    for(size_t i = 0; i < input.size(); i++) {
        output.push_back(input[i]);
    }
    
    return output;
}

