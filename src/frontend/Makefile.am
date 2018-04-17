AM_CPPFLAGS = -I$(srcdir)/../modules
AM_CXXFLAGS = $(PICKY_CXXFLAGS) $(CXX14_FLAGS)

bin_PROGRAMS = nn
nn_SOURCES = main.cc 
nn_LDADD = -L../modules/ -lnnfc ../modules/libnnfc.la -lnn ../modules/libnn.a
