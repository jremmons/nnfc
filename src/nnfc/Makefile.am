AM_CPPFLAGS = $(CXX14_FLAGS) $(JPEG_CFLAGS) \
              $(EIGEN3_CFLAGS) $(EIGEN3_UNSUPPORTED_CFLAGS) \
              -I$(srcdir)/../nn/

AM_CXXFLAGS = $(PICKY_CXXFLAGS) -fpic -L$(srcdir)/../nn \
              -Wl,-Bstatic -lnn \
              -Wl,-Bdynamic -lturbojpeg -ljpeg

lib_LTLIBRARIES = libnnfc.la

libnnfc_la_SOURCES = nnfc_CXXAPI.hh nnfc_CXXAPI.cc \
                     noop_codec.hh noop_codec.cc \
                     jpeg_codec.hh jpeg_codec.cc \
                     nnfc1_codec.hh nnfc1_codec.cc

libnnfc_la_LDFLAGS = -version-info 0:0:0
