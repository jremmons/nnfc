AM_CPPFLAGS = $(CXX14_FLAGS) $(JPEG_INCLUDE) \
              $(EIGEN3_CFLAGS) $(EIGEN3_UNSUPPORTED_CFLAGS) \
              $(HDF5_CFLAGS) $(HDF5_CPPFLAGS) \
              -I$(srcdir)/../src/nn -I$(srcdir)/../src/nnfc

AM_CXXFLAGS = $(PICKY_CXXFLAGS) $(HDF5_LDFLAGS) $(HDF5_LIBS) \
              -L$(srcdir)/../src/nn -lnn

check_PROGRAMS = avgpool.bin batchnorm.bin conv2d.bin fc.bin relu.bin simplecnn.bin simplecnn_hl.bin #resnet18.bin

avgpool_bin_SOURCES = avgpool_test.cc
avgpool_bin_LDADD =

batchnorm_bin_SOURCES = batchnorm_test.cc
batchnorm_bin_LDADD =

conv2d_bin_SOURCES = conv2d_test.cc
conv2d_bin_LDADD =

fc_bin_SOURCES = fc_test.cc
fc_bin_LDADD =

relu_bin_SOURCES = relu_test.cc
relu_bin_LDADD =

simplecnn_bin_SOURCES = simplecnn_test.cc
simplecnn_bin_LDADD =

simplecnn_hl_bin_SOURCES = simplecnn_hl_test.cc
simplecnn_hl_bin_LDADD =

dist_check_SCRIPTS = pythonpath_python.test \
	                 import_python.test \
                     noop_python.test \
                     nnfc_python.test \
                     gpu2cpu_python.test \
                     avgpool_cpp.test \
                     batchnorm_cpp.test \
                     convolution_cpp.test \
                     fc_cpp.test \
                     relu_cpp.test \
                     simplecnn_cpp.test \
                     simplecnn_hl_cpp.test

AM_TESTS_ENVIRONMENT = ./test-environment.sh

TESTS = $(dist_check_SCRIPTS)
