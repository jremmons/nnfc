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

  std::ifstream t(argv[1]);
  std::string str((std::istreambuf_iterator<char>(t)),
                  std::istreambuf_iterator<char>());

  // std::cout << "input text: " << str << std::endl;
  std::vector<char> uncompressed_input(str.begin(), str.end());
  std::cout << "input size: " << uncompressed_input.size() << std::endl;

  std::string base64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  
  codec::ArithmeticEncoder<codec::SimpleAdaptiveModel> encoder(64);
  for (auto c : uncompressed_input) {
  
    uint32_t sym = 0xDEADBEEF;
    for (size_t sym_ = 0; sym_ < base64.length(); sym_++) {
        if (c == base64[sym_]) {
            sym = sym_;
            break;
        }
    }

    assert(sym != 0xDEADBEEF);
    encoder.encode_symbol(sym);
  }
  const std::vector<char> compressed_output = encoder.finish();
  std::cout << "compressed size (bytes): " << compressed_output.size()
            << std::endl;

  std::ofstream output_file(argv[2], std::ios::out | std::ios::binary);
  output_file.write(compressed_output.data(), compressed_output.size());
}
