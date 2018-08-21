#include <iostream>

#include <fstream>
#include <streambuf>
#include <string>

#include "arithmetic_coder.hh"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "usage: " << argv[0] << " <input_file> <output_file>\n";
    return -1;
  }

  std::ifstream input_file(argv[1], std::ios::in | std::ios::binary);
  std::vector<char> compressed_input(std::istreambuf_iterator<char>{input_file},
                                     {});

  codec::ArithmeticDecoder<codec::SimpleAdaptiveModel> decoder(compressed_input, 64);

  std::string base64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::vector<char> uncompressed_output;
  while (not decoder.done()) {
    uint32_t symbol = decoder.decode_symbol();

    // end of text symbol
    if (symbol == 64) {
        break; 
    }

    uncompressed_output.push_back(base64[symbol]);
    
  }

  std::cout << "uncompressed size: " << uncompressed_output.size() << std::endl;

  std::ofstream output_file(argv[2], std::ios::out | std::ios::binary);
  output_file.write(uncompressed_output.data(), uncompressed_output.size());
}
