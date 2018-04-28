AM_CPPFLAGS = $(CXX14_FLAGS) $(JPEG_INCLUDE) \
              $(EIGEN3_CFLAGS) $(EIGEN3_UNSUPPORTED_CFLAGS) \
              $(HDF5_CFLAGS) $(HDF5_CPPFLAGS) -I../src/modules
AM_CXXFLAGS = $(PICKY_CXXFLAGS) 

check_PROGRAMS = avgpool.bin batchnorm.bin conv2d.bin fc.bin relu.bin simplecnn.bin #resnet18.bin

avgpool_bin_SOURCES = avgpool_test.cc
avgpool_bin_LDADD = -L../src/modules -lnn ../src/modules/libnn.a $(HDF5_LDFLAGS) $(HDF5_LIBS)

batchnorm_bin_SOURCES = batchnorm_test.cc
batchnorm_bin_LDADD = -L../src/modules -lnn ../src/modules/libnn.a $(HDF5_LDFLAGS) $(HDF5_LIBS)

conv2d_bin_SOURCES = conv2d_test.cc
conv2d_bin_LDADD = -L../src/modules -lnn ../src/modules/libnn.a $(HDF5_LDFLAGS) $(HDF5_LIBS)

fc_bin_SOURCES = fc_test.cc
fc_bin_LDADD = -L../src/modules -lnn ../src/modules/libnn.a $(HDF5_LDFLAGS) $(HDF5_LIBS)

relu_bin_SOURCES = relu_test.cc
relu_bin_LDADD = -L../src/modules -lnn ../src/modules/libnn.a $(HDF5_LDFLAGS) $(HDF5_LIBS)

simplecnn_bin_SOURCES = simplecnn_test.cc
simplecnn_bin_LDADD = -L../src/modules -lnn ../src/modules/libnn.a $(HDF5_LDFLAGS) $(HDF5_LIBS)

#resnet18_bin_SOURCES = resnet18_test.cc
#resnet18_bin_LDADD = -L ../src/modules -lnn ../src/modules/libnn.a $(HDF5_LDFLAGS) $(HDF5_LIBS)

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
                     simplecnn_cpp.test
                     #resnet18_cpp.test

AM_TESTS_ENVIRONMENT = ./test-environment.sh

TESTS = $(dist_check_SCRIPTS)
