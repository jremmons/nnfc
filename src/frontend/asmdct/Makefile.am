AM_CPPFLAGS = -I$(srcdir)/../../codec -I$(srcdir)/../../ \
              $(EIGEN3_CFLAGS) $(EIGEN3_UNSUPPORTED_CFLAGS) 

AM_CXXFLAGS = $(PICKY_CXXFLAGS) $(CXX14_FLAGS) \
              -L$(srcdir)/../../codec/ -lcodec \
              $(FFTW3F_LIBS)

bin_PROGRAMS = do_dct

do_dct_SOURCES = do_dct.cc 
do_dct_LDADD = 
