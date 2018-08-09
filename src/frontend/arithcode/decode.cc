#include <iostream>

#include <fstream>
#include <streambuf>
#include <string>

#include "arithmetic_encoder.hh"

int main(int argc, char* argv[]) {
  if (argc != 3) {
      std::cout << "usage: " << argv[0] << " <input_file> <output_file>\n";
      return -1;
  }

  std::ifstream input_file(argv[1], std::ios::in | std::ios::binary);
  std::vector<char> compressed_input(std::istreambuf_iterator<char>{input_file},
                                     {});

  const std::vector<char> uncompressed_output = codec::arith_decode(compressed_input);
  std::cout << "uncompressed size: " << uncompressed_output.size() << std::endl;

  std::ofstream output_file(argv[2], std::ios::out | std::ios::binary);
  output_file.write(uncompressed_output.data(), uncompressed_output.size());
}
