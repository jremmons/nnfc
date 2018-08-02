AM_CPPFLAGS = -I$(srcdir)/../../nn -I$(srcdir)/../../nnfc \
              $(JPEG_CFLAGS) \
              $(HDF5_CFLAGS) $(HDF5_CPPFLAGS) \
              $(EIGEN3_CFLAGS) $(EIGEN3_UNSUPPORTED_CFLAGS) 

AM_CXXFLAGS = $(PICKY_CXXFLAGS) $(CXX14_FLAGS) \
              $(HDF5_LDFLAGS) $(HDF5_LIBS) \
              -L$(srcdir)/../../nn/ -lnn \
              -L$(srcdir)/../../nnfc/ -L$(srcdir)/../../nnfc/.libs

bin_PROGRAMS = simplenet9

#nn_SOURCES = main.cc 
#nn_LDADD = $(srcdir)/../nnfc/libnnfc.la

simplenet9_SOURCES = simplenet9.cc 
simplenet9_LDADD = $(srcdir)/../../nnfc/libnnfc.la -lturbojpeg
