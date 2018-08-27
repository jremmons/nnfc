SUBDIRS = arithcode

AM_CPPFLAGS = $(CXX14_FLAGS) $(THIRD_PARTY_CFLAGS) \
              $(JPEG_INCLUDE) \
              $(EIGEN3_CFLAGS) $(EIGEN3_UNSUPPORTED_CFLAGS) \
              $(HDF5_CFLAGS) $(HDF5_CPPFLAGS) \
              -I$(srcdir)/../src/ \
              -I$(srcdir)/../src/nn \
              -I$(srcdir)/../src/nnfc

AM_CXXFLAGS = $(PICKY_CXXFLAGS)

AM_LDFLAGS = -L$(srcdir)/../src/nnfc/.libs \
             -L$(srcdir)/../src/nn \
             -L$(srcdir)/../src/codec

LDADD = -lnnfc -l:libcodec.a -l:libnn.a $(HDF5_LDFLAGS) $(HDF5_LIBS)

check_PROGRAMS = avgpool.bin \
                 batchnorm.bin batchnorm_hl.bin \
                 conv2d.bin conv2d_hl.bin \
                 fc.bin \
                 relu.bin relu_hl.bin \
                 composed_hl.bin \
                 simplecnn.bin simplecnn_hl.bin \
                 cxxapi_simple.bin

avgpool_bin_SOURCES = avgpool_test.cc

batchnorm_bin_SOURCES = batchnorm_test.cc

batchnorm_hl_bin_SOURCES = batchnorm_hl_test.cc

conv2d_bin_SOURCES = conv2d_test.cc

conv2d_hl_bin_SOURCES = conv2d_hl_test.cc

fc_bin_SOURCES = fc_test.cc

relu_bin_SOURCES = relu_test.cc

relu_hl_bin_SOURCES = relu_hl_test.cc

composed_hl_bin_SOURCES = composed_hl_test.cc

simplecnn_bin_SOURCES = simplecnn_test.cc

simplecnn_hl_bin_SOURCES = simplecnn_hl_test.cc

cxxapi_simple_bin_SOURCES = cxxapi_simple.cc

dist_check_SCRIPTS = pythonpath_python.test \
                     import_python.test \
                     noop_python.test \
                     nnfc_python.test \
                     jpeg_python.test \
                     nnfc_codecs_python.test \
                     avgpool_cpp.test \
                     batchnorm_cpp.test \
                     batchnorm_hl_cpp.test \
                     convolution_cpp.test \
                     convolution_hl_cpp.test \
                     fc_cpp.test \
                     relu_cpp.test \
                     relu_hl_cpp.test \
                     composed_hl_cpp.test \
                     simplecnn_hl_cpp.test \
                     arithcode/test_arithmetic_coder.test

AM_TESTS_ENVIRONMENT = ./test-environment.sh

TESTS = $(dist_check_SCRIPTS) \
        ./cxxapi_simple.bin
