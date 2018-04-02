AM_CPPFLAGS = I$(srcdir)/../../third_party/jpeg/libjpeg-turbo.compiled/include
AM_CXXFLAGS = $(PICKY_CXXFLAGS) $(CXX14_FLAGS) -fPIC

noinst_LIBRARIES = libnnfc.a libnoop.a 

libnnfc_a_SOURCES = nnfc.hh nnfc.cc blob4d.hh blob1d.hh
libnnfc_a_LADD = ../../third_party/jpeg/libjpeg-turbo.compiled/lib/libturbojpeg.a -lturbojpeg

libnoop_a_SOURCES = noop.hh noop.cc blob4d.hh blob1d.hh


