AM_CPPFLAGS = $(CXX14_FLAGS) $(THIRD_PARTY_CFLAGS) \
              $(EIGEN3_CFLAGS) $(EIGEN3_UNSUPPORTED_CFLAGS)  \
              -I$(srcdir)/../../

AM_CXXFLAGS = $(PICKY_CXXFLAGS) 

bin_PROGRAMS = do_dct

do_dct_SOURCES = do_dct.cc 
do_dct_LDADD = $(srcdir)/../../codec/libcodec.a
