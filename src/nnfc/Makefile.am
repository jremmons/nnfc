AM_CPPFLAGS = $(CXX14_FLAGS) $(JPEG_CFLAGS) \
              $(EIGEN3_CFLAGS) $(EIGEN3_UNSUPPORTED_CFLAGS) \
              -I$(srcdir)/../nn/

AM_CXXFLAGS = $(PICKY_CXXFLAGS) -fpic -L$(srcdir)/../nn -lnn

lib_LTLIBRARIES = libnnfc.la 

libnnfc_la_SOURCES = nnfc.hh nnfc.cc blob4d.hh blob1d.hh
libnnfc_la_LDFLAGS = -version-info 0:0:0 