#include "mpeg.hh"

extern "C" {
  #include <libavcodec/avcodec.h>
  #include <libavutil/opt.h>
}
#include <memory>
#include <stdexcept>
#include <iostream>

using namespace std;
using namespace codec;

template <class T>
inline T* CheckAVCommand(const char* s_attempt, T* return_value) {
  if (return_value) {
    return return_value;
  }

  throw runtime_error(s_attempt);
}

inline int CheckAVCommand(const char* s_attempt, const int return_value) {
  if (return_value >= 0) {
    return return_value;
  }

  throw runtime_error(s_attempt);
}

template<AVCodecID codec_id>
vector<uint8_t> MPEGEncoder<codec_id>::encode(const vector<uint8_t>& image,
                                              const size_t width,
                                              const size_t height,
                                              const size_t channels) {
  if (channels != 1 and channels != 3) {
    throw runtime_error("number of channels must be 1 or 3");
  }

  if ((channels == 1 and image.size() < width * height) or
      (channels == 3 and image.size() < width * height * 3 / 2)) {
    throw runtime_error("unexpected image length");
  }

  /* Preparing the codec, the context and the frame */
  av_log_set_level(AV_LOG_QUIET);
  avcodec_register_all();
  AVCodec* encoder =
      CheckAVCommand("find_encoder", avcodec_find_encoder(codec_id_));

  shared_ptr<AVCodecContext> context{
      CheckAVCommand("alloc_context", avcodec_alloc_context3(encoder)),
      [](auto* c) { avcodec_free_context(&c); }};

  context->pix_fmt = pix_fmt_;
  context->width = width;
  context->height = height;
  context->bit_rate = 1 << 10;
  context->bit_rate_tolerance = 0;
  context->time_base = (AVRational){1, 20};
  context->framerate = (AVRational){60, 1};
  context->gop_size = 0;
  context->max_b_frames = 0;
  context->qmin = quantizer_;
  context->qmax = quantizer_;
  context->qcompress = 0.5;
  av_opt_set(context->priv_data, "tune", "zerolatency", 0);
  av_opt_set(context->priv_data, "preset", "fast", 0);

  CheckAVCommand("avcodec_open2", avcodec_open2(context.get(), encoder, NULL));

  shared_ptr<AVFrame> frame{
      CheckAVCommand("encoder_frame", av_frame_alloc()),
      [](auto* f) { av_frame_free(&f); }};

  frame->width = width;
  frame->height = height;
  frame->format = pix_fmt_;
  frame->pts = 0;

  shared_ptr<AVPacket> packet{
      CheckAVCommand("encoder_frame", av_packet_alloc()),
      [](auto* p) { av_packet_free(&p); }};

  CheckAVCommand("av_frame_get_buffer", av_frame_get_buffer(frame.get(), 32));
  CheckAVCommand("av_frame_make_writable", av_frame_make_writable(frame.get()));

  for (size_t row = 0; row < height; row++) {
    memcpy(frame->data[0] + frame->linesize[0] * row,
           image.data() + width * row, width);
  }

  if (channels == 1) {
    memset(frame->data[1], 0, frame->linesize[1] * height / 2);
    memset(frame->data[2], 0, frame->linesize[2] * height / 2);
  }
  else {
    for (size_t row = 0; row < height / 2; row++) {
      memcpy(frame->data[1] + frame->linesize[1] * row,
             image.data() + width * height + width * row / 2,
             width / 2);

      memcpy(frame->data[2] + frame->linesize[2] * row,
             image.data() + 5 * width * height / 4 + width * row / 2,
             width / 2);
    }
  }

  vector<uint8_t> result;
  int ret = CheckAVCommand("avcodec_send_frame", avcodec_send_frame(context.get(), frame.get()));

  while (ret >= 0) {
    ret = avcodec_receive_packet(context.get(), packet.get());
    if (ret == AVERROR(EAGAIN) or ret == AVERROR_EOF) {
      break;
    }
    else if (ret < 0) {
      throw runtime_error("receive_packet");
    }

    size_t start_idx = result.size();
    result.resize(result.size() + packet->size);
    memcpy(result.data() + start_idx, packet->data, packet->size);
  }

  return result;
}

vector<vector<uint8_t>> decode_frame(AVCodecContext * context, AVFrame * frame,
                                     AVPacket * packet) {
  vector<vector<uint8_t>> outputs;

  size_t width = frame->width;
  size_t height = frame->height;
  int ret = CheckAVCommand("send_packet", avcodec_send_packet(context, packet));

  while (ret >= 0) {
    ret = avcodec_receive_frame(context, frame);
    if (ret == AVERROR(EAGAIN) or ret == AVERROR_EOF) {
      return outputs;
    }
    else if (ret < 0) {
      throw runtime_error("receive_frame");
    }

    vector<uint8_t> output;
    output.resize(width * height * 3 / 2);

    for (size_t row = 0; row < height; row++) {
      memcpy(output.data() + width * row,
             frame->data[0] + frame->linesize[0] * row,
             width);
    }

    for (size_t row = 0; row < height / 2; row++) {
      memcpy(output.data() + width * height + width * row / 2,
             frame->data[1] + frame->linesize[1] * row,
             width / 2);

      memcpy(output.data() + 5 * width * height / 4 + width * row / 2,
             frame->data[2] + frame->linesize[2] * row,
             width / 2);
    }

    outputs.emplace_back(move(output));
  }

  return outputs;
}

template<AVCodecID codec_id>
vector<uint8_t> MPEGDecoder<codec_id>::decode(const vector<uint8_t> & compressed,
                                           const size_t width,
                                           const size_t height) {
  //compressed.resize(compressed.size() + AV_INPUT_BUFFER_PADDING_SIZE, 0);

  av_log_set_level(AV_LOG_QUIET);
  avcodec_register_all();
  AVCodec* decoder =
      CheckAVCommand("find_decoder", avcodec_find_decoder(codec_id_));

  shared_ptr<AVCodecContext> context{
      CheckAVCommand("alloc_context", avcodec_alloc_context3(decoder)),
      [](auto* c) { avcodec_free_context(&c); }};

  context->pix_fmt = pix_fmt_;
  context->width = width;
  context->height = height;
  av_opt_set(context->priv_data, "tune", "zerolatency", 0);

  shared_ptr<AVCodecParserContext> parser{
      CheckAVCommand("av_parser_init", av_parser_init(decoder->id)),
      [](auto* c) { av_parser_close(c); }};

  parser->flags |= PARSER_FLAG_COMPLETE_FRAMES;

  CheckAVCommand("avcodec_open2", avcodec_open2(context.get(), decoder, NULL));

  shared_ptr<AVFrame> frame{
      CheckAVCommand("frame_alloc", av_frame_alloc()),
      [](auto* f) { av_frame_free(&f); }};

  frame->width = width;
  frame->height = height;
  frame->format = pix_fmt_;
  frame->pts = 0;

  CheckAVCommand("av_frame_get_buffer", av_frame_get_buffer(frame.get(), 32));
  CheckAVCommand("av_frame_make_writable", av_frame_make_writable(frame.get()));

  AVPacket packet;
  av_init_packet(&packet);

  const uint8_t * dataptr = compressed.data();
  int size = compressed.size();

  vector<vector<uint8_t>> outputs;

  while (size > 0) {
    int ret = CheckAVCommand("parser_parse", av_parser_parse2(parser.get(),
                                                              context.get(),
                                                              &packet.data,
                                                              &packet.size,
                                                              dataptr, size,
                                                              AV_NOPTS_VALUE,
                                                              AV_NOPTS_VALUE,
                                                              0));

    dataptr += ret;
    size -= ret;

    if (packet.size > 0) {
      outputs = decode_frame(context.get(), frame.get(), &packet);
    }
  }

  /* flushing the decoder */
  auto flushed_outputs = decode_frame(context.get(), frame.get(), nullptr);
  outputs.insert(outputs.end(), make_move_iterator(flushed_outputs.begin()),
                 make_move_iterator(flushed_outputs.end()));

  if (outputs.size() != 1) {
    throw runtime_error("unexpected number of outputs");
  }

  return outputs.at(0);
}

template class MPEGEncoder<AV_CODEC_ID_H264>;
template class MPEGDecoder<AV_CODEC_ID_H264>;
template class MPEGEncoder<AV_CODEC_ID_H265>;
template class MPEGDecoder<AV_CODEC_ID_H265>;
