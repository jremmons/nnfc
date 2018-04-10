AM_CPPFLAGS = $(CXX14_FLAGS) $(JPEG_INCLUDES)
AM_CXXFLAGS = $(PICKY_CXXFLAGS) -fPIC

noinst_LIBRARIES = libnnfc.a libnoop.a libnn.a

libnnfc_a_SOURCES = nnfc.hh nnfc.cc blob4d.hh blob1d.hh

libnoop_a_SOURCES = noop.hh noop.cc blob4d.hh blob1d.hh

libnn_a_SOURCES = nn.hh nn.cc \
                  activation.hh activation.cc \
                  convolution.hh convolution.cc \
                  fullyconnected.hh fullyconnected.cc \
                  pool.hh pool.cc \
                  normalization.hh normalization.cc 
