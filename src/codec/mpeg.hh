#ifndef _CODEC_MPEG_HH
#define _CODEC_MPEG_HH

extern "C" {
  #include <libavcodec/avcodec.h>
}
#include <cstdint>
#include <vector>

namespace codec {

template<AVCodecID codec_id>
class MPEGEncoder {
private:
  const AVCodecID codec_id_{codec_id};
  const AVPixelFormat pix_fmt_{AV_PIX_FMT_YUV420P};

  const int quantizer_;

public:
  MPEGEncoder(const int quantizer) : quantizer_(quantizer) {}

  std::vector<uint8_t> encode(const std::vector<uint8_t>& image, const size_t width,
                              const size_t height, const size_t channels);
};

template<AVCodecID codec_id>
class MPEGDecoder {
private:
  const AVCodecID codec_id_{codec_id};
  const AVPixelFormat pix_fmt_{AV_PIX_FMT_YUV420P};

public:
  MPEGDecoder() {}

  std::vector<uint8_t> decode(const std::vector<uint8_t>& coded_bitstream,
                              const size_t width, const size_t height);
};

using AVCEncoder = MPEGEncoder<AV_CODEC_ID_H264>;
using AVCDecoder = MPEGDecoder<AV_CODEC_ID_H264>;
using HEIFEncoder = MPEGEncoder<AV_CODEC_ID_H265>;
using HEIFDecoder = MPEGDecoder<AV_CODEC_ID_H265>;

}  // namespace codec

#endif /* _CODEC_MPEG_HH */
