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

  std::ifstream t(argv[1]);
  std::string str((std::istreambuf_iterator<char>(t)),
                  std::istreambuf_iterator<char>());

  std::cout << "input text: " << str << std::endl;
  std::vector<char> uncompressed_input(str.begin(), str.end());
  std::cout << "input size: " << uncompressed_input.size() << std::endl;

  const std::vector<char> compressed_output =
      codec::arith_encode(uncompressed_input);
  std::cout << "compressed size (bytes): " << compressed_output.size() << std::endl;

  std::ofstream output_file(argv[2], std::ios::out | std::ios::binary);
  output_file.write(compressed_output.data(), compressed_output.size());
}
