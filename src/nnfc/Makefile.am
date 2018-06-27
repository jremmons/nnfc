AM_CPPFLAGS = $(CXX14_FLAGS) $(JPEG_CFLAGS) \
              $(EIGEN3_CFLAGS) $(EIGEN3_UNSUPPORTED_CFLAGS) \
              $(AVFORMAT_CFLAGS) $(AVCODEC_CFLAGS) $(AVUTIL_CFLAGS) \
              $(SWSCALE_CFLAGS) \
              -I$(srcdir)/../nn/ -I$(srcdir)/..

AM_CXXFLAGS = $(PICKY_CXXFLAGS) -fPIC \
              -Wl,-Bstatic -L$(srcdir)/../nn \
              -L$(srcdir)/../codec -l:libnn.a -l:libcodec.a \
              -Wl,-Bdynamic -lturbojpeg -Wl,-Bdynamic -ljpeg \
              $(AVFORMAT_LIBS) $(AVCODEC_LIBS) $(AVUTIL_LIBS) \
              $(SWSCALE_LIBS) -Wl,-Bdynamic -lswscale

lib_LTLIBRARIES = libnnfc.la

EXTRA_libnnfc_la_DEPENDENCIES = ../codec/libcodec.a ../nn/libnn.a

libnnfc_la_SOURCES = nnfc_CXXAPI.hh nnfc_CXXAPI.cc \
                     noop_codec.hh noop_codec.cc \
                     swizzler.hh swizzler.cc \
                     jpeg_codec.hh jpeg_codec.cc \
                     jpeg_image_codec.hh jpeg_image_codec.cc \
                     mpeg_image_codec.hh mpeg_image_codec.cc \
                     mpeg_codec.hh mpeg_codec.cc \
                     nnfc1_codec.hh nnfc1_codec.cc

libnnfc_la_LDFLAGS = -version-info 0:0:0
