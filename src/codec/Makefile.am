AM_CPPFLAGS = $(CXX14_FLAGS) $(JPEG_CFLAGS) \
              $(EIGEN3_CFLAGS) $(EIGEN3_UNSUPPORTED_CFLAGS) \
              -I$(srcdir)/..

AM_CXXFLAGS = $(PICKY_CXXFLAGS) -fPIC

noinst_LIBRARIES = libcodec.a

libcodec_a_SOURCES = jpeg.hh jpeg.cc \
                     mpeg.hh mpeg.cc \
                     swizzle.hh swizzle.cc \
                     utils.hh utils.cc
