AM_CPPFLAGS = -I$(srcdir)/../modules \
              $(EIGEN3_CFLAGS) $(EIGEN3_UNSUPPORTED_CFLAGS)

AM_CXXFLAGS = $(PICKY_CXXFLAGS) $(CXX14_FLAGS)

bin_PROGRAMS = nn
nn_SOURCES = main.cc 
nn_LDADD = -L../modules/ -lnnfc ../modules/libnnfc.la -lnn ../modules/libnn.a
