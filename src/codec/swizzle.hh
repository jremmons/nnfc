#ifndef _CODEC_SIWZZLE_HH
#define _CODEC_SIWZZLE_HH

#include <memory>
#include <vector>

extern "C" {
#include <libswscale/swscale.h>
}

namespace codec {

class RGBp_to_YUV420p {
 private:
  std::shared_ptr<SwsContext> context_;

  const int height_;
  const int width_;

 public:
  RGBp_to_YUV420p(size_t height, size_t width);
  ~RGBp_to_YUV420p();
  std::vector<uint8_t> convert(std::vector<uint8_t>& src_frame);
};

class YUV420p_to_RGBp {
 private:
  std::shared_ptr<SwsContext> context_;

  const int height_;
  const int width_;

 public:
  YUV420p_to_RGBp(size_t height, size_t width);
  ~YUV420p_to_RGBp();
  std::vector<uint8_t> convert(std::vector<uint8_t>& src_frame);
};

class RGB24_to_YUV422p {
 private:
  std::shared_ptr<SwsContext> context_;

  const int height_;
  const int width_;

 public:
  RGB24_to_YUV422p(size_t height, size_t width);
  ~RGB24_to_YUV422p();
  std::vector<uint8_t> convert(std::vector<uint8_t>& src_frame);
};

class YUV422p_to_RGB24 {
 private:
  std::shared_ptr<SwsContext> context_;

  const int height_;
  const int width_;

 public:
  YUV422p_to_RGB24(size_t height, size_t width);
  ~YUV422p_to_RGB24();
  std::vector<uint8_t> convert(std::vector<uint8_t>& src_frame);
};
}

#endif  // _CODEC_SIWZZLE_HH
