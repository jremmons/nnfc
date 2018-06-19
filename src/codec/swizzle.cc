#include <cassert>
#include <memory>
#include <vector>
#include <iostream>

#include "swizzle.hh"

extern "C" {
#include <libswscale/swscale.h>
}

codec::RGBp_to_YUV420p::RGBp_to_YUV420p(size_t height, size_t width) :
    context_(sws_getContext(width, height,
                            AV_PIX_FMT_GBRP,
                            width, height,
                            AV_PIX_FMT_YUV420P,
                            0, nullptr, nullptr, nullptr),
             [](SwsContext* ptr) { sws_freeContext(ptr); }),
    height_(height),
    width_(width)
{
    av_log_set_level(AV_LOG_QUIET);
}

codec::RGBp_to_YUV420p::~RGBp_to_YUV420p() {}

std::vector<uint8_t> codec::RGBp_to_YUV420p::convert(std::vector<uint8_t>& src_frame) {

    assert(src_frame.size() == static_cast<size_t>(3*height_*width_));
    
    uint8_t *src_data[3] = {
        src_frame.data() + height_*width_,
        src_frame.data() + 2*height_*width_,
        src_frame.data(),
    };
    int src_linesize[3] = { width_, width_, width_ };

    std::vector<uint8_t> dest_frame(3*(height_*width_/2));
    uint8_t *dest_data[3] = {
        dest_frame.data(),
        dest_frame.data() + height_*width_,
        dest_frame.data() + (height_*width_) + (height_*width_/4),
    };
    int dest_linesize[3] = { width_, width_/2, width_/2 };
    
    sws_scale(context_.get(), src_data, src_linesize, 0, height_, dest_data, dest_linesize);
      
    return dest_frame;
}

codec::YUV420p_to_RGBp::YUV420p_to_RGBp(size_t height, size_t width) :
    context_(sws_getContext(width, height,
                            AV_PIX_FMT_YUV420P,
                            width, height,
                            AV_PIX_FMT_GBRP,
                            0, nullptr, nullptr, nullptr),
             [](SwsContext* ptr) { sws_freeContext(ptr); }),
    height_(height),
    width_(width)
{
    av_log_set_level(AV_LOG_QUIET);
}

codec::YUV420p_to_RGBp::~YUV420p_to_RGBp() {}

std::vector<uint8_t> codec::YUV420p_to_RGBp::convert(std::vector<uint8_t>& src_frame) {

    assert(src_frame.size() == static_cast<size_t>(3*(height_*width_/2)));

    uint8_t *src_data[3] = {
        src_frame.data(),
        src_frame.data() + height_*width_,
        src_frame.data() + (height_*width_) + (height_*width_/4),
    };
    int src_linesize[3] = { width_, width_/2, width_/2 };

    std::vector<uint8_t> dest_frame(3*height_*width_);
    uint8_t *dest_data[3] = {
        dest_frame.data() + height_*width_,
        dest_frame.data() + 2*height_*width_,
        dest_frame.data(),
    };
    int dest_linesize[3] = { width_, width_, width_ };
    
    sws_scale(context_.get(), src_data, src_linesize, 0, height_, dest_data, dest_linesize);
      
    return dest_frame;
}

//////////////////////////////////////////////////////////////////////

codec::RGB24_to_YUV422p::RGB24_to_YUV422p(size_t height, size_t width) :
    context_(sws_getContext(width, height,
                            AV_PIX_FMT_RGB24,
                            width, height,
                            AV_PIX_FMT_YUV422P,
                            0, nullptr, nullptr, nullptr),
             [](SwsContext* ptr) { sws_freeContext(ptr); }),
    height_(height),
    width_(width)
{
    av_log_set_level(AV_LOG_QUIET);
}

codec::RGB24_to_YUV422p::~RGB24_to_YUV422p() {}

std::vector<uint8_t> codec::RGB24_to_YUV422p::convert(std::vector<uint8_t>& src_frame) {

    assert(src_frame.size() == static_cast<size_t>(3*height_*width_));

    uint8_t *src_data[1] = { src_frame.data() };
    int src_linesize[1] = { 3*width_ };

    std::vector<uint8_t> dest_frame(2*height_*width_);
    uint8_t *dest_data[3] = {
        dest_frame.data(),
        dest_frame.data() + height_*width_,
        dest_frame.data() + (height_*width_) + (height_*width_/2)
    };
    int dest_linesize[3] = { width_, width_/2, width_/2 };
    
    sws_scale(context_.get(), src_data, src_linesize, 0, height_, dest_data, dest_linesize);
      
    return dest_frame;
}


codec::YUV422p_to_RGB24::YUV422p_to_RGB24(size_t height, size_t width) :
    context_(sws_getContext(width, height,
                            AV_PIX_FMT_YUV422P,
                            width, height,
                            AV_PIX_FMT_RGB24,
                            0, nullptr, nullptr, nullptr),
             [](SwsContext* ptr) { sws_freeContext(ptr); }),
    height_(height),
    width_(width)
{
    av_log_set_level(AV_LOG_QUIET);
}

codec::YUV422p_to_RGB24::~YUV422p_to_RGB24() {}

std::vector<uint8_t> codec::YUV422p_to_RGB24::convert(std::vector<uint8_t>& src_frame) {

     assert(src_frame.size() == static_cast<size_t>(2*height_*width_));

    uint8_t *src_data[3] = {
        src_frame.data(),
        src_frame.data() + height_*width_,
        src_frame.data() + (height_*width_) + (height_*width_/2)
    };
    int src_linesize[3] = { width_, width_/2, width_/2 };

    std::vector<uint8_t> dest_frame(3*height_*width_);
    uint8_t *dest_data[1] = {
        dest_frame.data(),
    };
    int dest_linesize[1] = { 3*width_ };
    
    sws_scale(context_.get(), src_data, src_linesize, 0, height_, dest_data, dest_linesize);
      
    return dest_frame;
}
