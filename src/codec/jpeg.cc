#include "jpeg.hh"

#include <iostream>
#include <stdexcept>

extern "C" {
#include <jpeglib.h>
}
#include <turbojpeg.h>

using namespace std;

vector<uint8_t> codec::JPEGEncoder::encode(vector<uint8_t>& image,
                                           const size_t width,
                                           const size_t height,
                                           const size_t channels) {
  if (channels != 1 and channels != 3) {
    throw runtime_error("number of channels must be 1 or 3");
  }

  jpeg_compress_struct context;
  jpeg_error_mgr jerr;

  jpeg_create_compress(&context);
  context.in_color_space = (channels == 1) ? JCS_GRAYSCALE : JCS_RGB;
  jpeg_set_defaults(&context);
  jpeg_set_quality(&context, quality_, true);
  context.arith_code = true;
  context.err = jpeg_std_error(&jerr);
  context.dct_method = JDCT_FASTEST;

  long unsigned int jpeg_size = 0;
  unsigned char* compressed_image = nullptr;

  context.image_width = width;
  context.image_height = height;
  context.input_components = channels;

  const size_t row_stride = channels * width;

  jpeg_mem_dest(&context, &compressed_image, &jpeg_size);
  jpeg_start_compress(&context, true);

  while (context.next_scanline < context.image_height) {
    unsigned char* location = &image[context.next_scanline * row_stride];
    jpeg_write_scanlines(&context, &location, 1);
  }

  jpeg_finish_compress(&context);
  jpeg_destroy_compress(&context);

  vector<uint8_t> result{compressed_image, compressed_image + jpeg_size};
  free(compressed_image);
  return result;
}
