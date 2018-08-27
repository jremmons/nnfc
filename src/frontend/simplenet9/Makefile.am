AM_CPPFLAGS = $(CXX14_FLAGS) $(THIRD_PARTY_CFLAGS) \
              $(JPEG_CFLAGS) \
              $(HDF5_CFLAGS) $(HDF5_CPPFLAGS) \
              $(EIGEN3_CFLAGS) $(EIGEN3_UNSUPPORTED_CFLAGS) \
              -I$(srcdir)/../../

AM_CXXFLAGS = $(PICKY_CXXFLAGS) \
              $(HDF5_LDFLAGS) $(HDF5_LIBS) \
              -L$(srcdir)/../../nn/ -lnn

bin_PROGRAMS = simplenet9

#nn_SOURCES = main.cc 
#nn_LDADD = $(srcdir)/../nnfc/libnnfc.la

simplenet9_SOURCES = simplenet9.cc 
simplenet9_LDADD = $(srcdir)/../../nnfc/libnnfc.la -lturbojpeg
