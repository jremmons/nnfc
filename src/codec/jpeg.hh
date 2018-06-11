#ifndef _CODEC_JPEG_HH
#define _CODEC_JPEG_HH

#include <vector>
#include "nn/tensor.hh"

namespace codec {

class JPEGEncoder {
 private:
  const int quality_;

 public:
  JPEGEncoder(const int quality) : quality_(quality) {}

  std::vector<uint8_t> encode(std::vector<uint8_t>& image, const size_t width,
                              const size_t height, const size_t channels);
};

}

#endif /* _CODEC_JPEG_HH */
