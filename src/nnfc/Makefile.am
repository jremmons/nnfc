AM_CPPFLAGS = $(CXX14_FLAGS) $(THIRD_PARTY_CFLAGS) \
              $(JPEG_CFLAGS) \
              $(EIGEN3_CFLAGS) $(EIGEN3_UNSUPPORTED_CFLAGS) \
              $(AVFORMAT_CFLAGS) $(AVCODEC_CFLAGS) $(AVUTIL_CFLAGS) \
              $(SWSCALE_CFLAGS) \
              -I$(srcdir)/..

AM_CXXFLAGS = $(PICKY_CXXFLAGS) -fPIC \
              -L$(srcdir)/../nn -L$(srcdir)/../codec \
              -Wl,-Bstatic -l:libcodec.a \
              -Wl,-Bstatic -l:libnn.a \
              -Wl,-Bdynamic \
              -lturbojpeg -ljpeg \
              $(AVFORMAT_LIBS) $(AVCODEC_LIBS) $(AVUTIL_LIBS) \
              $(SWSCALE_LIBS) $(FFTW3F_LIBS)

lib_LTLIBRARIES = libnnfc.la

EXTRA_libnnfc_la_DEPENDENCIES = ../codec/libcodec.a ../nn/libnn.a

libnnfc_la_SOURCES = nnfc_CXXAPI.hh nnfc_CXXAPI.cc \
                     noop_codec.hh noop_codec.cc \
                     swizzler.hh swizzler.cc \
                     jpeg_codec.hh jpeg_codec.cc \
                     jpeg_image_codec.hh jpeg_image_codec.cc \
                     mpeg_image_codec.hh mpeg_image_codec.cc \
                     mpeg_codec.hh mpeg_codec.cc \
                     nnfc1_codec.hh nnfc1_codec.cc \
                     nnfc2_codec.hh nnfc2_codec.cc

libnnfc_la_LDFLAGS = -version-info 0:0:0
