AM_CPPFLAGS = -I$(srcdir)/../nn -I$(srcdir)/../nnfc \
              $(EIGEN3_CFLAGS) $(EIGEN3_UNSUPPORTED_CFLAGS) 

AM_CXXFLAGS = $(PICKY_CXXFLAGS) $(CXX14_FLAGS) \
              -L$(srcdir)/../nn/ -lnn \
              -L$(srcdir)/../nnfc/ -L$(srcdir)/../nnfc/.libs

bin_PROGRAMS = nn
nn_SOURCES = main.cc 
nn_LDADD = $(srcdir)/../nnfc/libnnfc.la
