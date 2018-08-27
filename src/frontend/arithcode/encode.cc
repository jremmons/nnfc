#include <iostream>

#include <fstream>
#include <streambuf>
#include <string>

#include "codec/arithmetic_coder.hh"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "usage: " << argv[0] << " <input_file> <output_file>\n";
    return -1;
  }

  std::ifstream t(argv[1]);
  std::string str((std::istreambuf_iterator<char>(t)),
                  std::istreambuf_iterator<char>());

  // std::cout << "input text: " << str << std::endl;
  std::vector<char> uncompressed_input(str.begin(), str.end());
  std::cout << "input size: " << uncompressed_input.size() << std::endl;

  // codec::ArithmeticEncoder<codec::SimpleModel> encoder;
  codec::ArithmeticEncoder<codec::SimpleAdaptiveModel> encoder(10);
  for (auto c : uncompressed_input) {
    // std::cout << c << "\n";
    uint32_t sym = 0;
    switch (c) {
      case 'A':
        sym = 0;
        break;
      case 'B':
        sym = 1;
        break;
      default:
        throw std::runtime_error("unrecognized symbol");
    }
    assert(sym == 0 or sym == 1);

    encoder.encode_symbol(sym);
  }
  const std::vector<char> compressed_output = encoder.finish();
  std::cout << "compressed size (bytes): " << compressed_output.size()
            << std::endl;

  std::ofstream output_file(argv[2], std::ios::out | std::ios::binary);
  output_file.write(compressed_output.data(), compressed_output.size());
}
