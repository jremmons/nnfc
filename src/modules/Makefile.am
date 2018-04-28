AM_CPPFLAGS = $(CXX14_FLAGS) $(JPEG_CFLAGS) \
              $(EIGEN3_CFLAGS) $(EIGEN3_UNSUPPORTED_CFLAGS)

AM_CXXFLAGS = $(PICKY_CXXFLAGS) -fpic

lib_LTLIBRARIES = libnnfc.la 

noinst_LIBRARIES =  libnn.a

libnnfc_la_SOURCES = nnfc.hh nnfc.cc blob4d.hh blob1d.hh
libnnfc_la_LDFLAGS = -version-info 0:0:0

libnn_a_SOURCES = nn.hh nn.cc \
                  activation.hh activation.cc \
                  convolution.hh convolution.cc \
                  fullyconnected.hh fullyconnected.cc \
                  pool.hh pool.cc \
                  normalization.hh normalization.cc 
