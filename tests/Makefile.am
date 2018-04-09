AM_CPPFLAGS = $(CXX14_FLAGS) $(JPEG_INCLUDE) $(HDF5_CFLAGS) $(HDF5_CPPFLAGS) -I../src/modules
AM_CXXFLAGS = $(PICKY_CXXFLAGS) 

check_PROGRAMS = conv2d.bin relu.bin

conv2d_bin_SOURCES = conv2d_test.cc
conv2d_bin_LDADD = -L ../src/modules -lnn ../src/modules/libnn.a $(HDF5_LDFLAGS) $(HDF5_LIBS)

relu_bin_SOURCES = relu_test.cc
relu_bin_LDADD = -L ../src/modules -lnn ../src/modules/libnn.a $(HDF5_LDFLAGS) $(HDF5_LIBS)

dist_check_SCRIPTS = pythonpath_python.test \
	                 import_python.test \
                     noop_python.test \
                     nnfc_python.test \
                     convolution_cpp.test \
                     relu_cpp.test

AM_TESTS_ENVIRONMENT = ./test-environment.sh

TESTS = $(dist_check_SCRIPTS)
