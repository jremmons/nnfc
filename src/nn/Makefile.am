AM_CPPFLAGS = $(CXX14_FLAGS) $(JPEG_CFLAGS) \
              $(EIGEN3_CFLAGS) $(EIGEN3_UNSUPPORTED_CFLAGS)

AM_CXXFLAGS = $(PICKY_CXXFLAGS) -fpic

noinst_LIBRARIES =  libnn.a

libnn_a_SOURCES = activation.hh activation.cc \
                  convolution.hh convolution.cc \
                  fullyconnected.hh fullyconnected.cc \
                  pool.hh pool.cc \
                  normalization.hh normalization.cc \
                  layers.hh layers.cc \
                  net.hh net.cc
