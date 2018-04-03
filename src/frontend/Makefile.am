AM_CPPFLAGS = -I$(srcdir)/../modules
AM_CXXFLAGS = $(PICKY_CXXFLAGS) $(CXX14_FLAGS)

bin_PROGRAMS = nn
nn_SOURCES = main.cc nn.hh nn.cc
nn_LDADD = -L../modules/ -lnnfc ../modules/libnnfc.a
