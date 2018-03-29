AM_CPPFLAGS = -I$(srcdir)/../modules
AM_CXXFLAGS = $(PICKY_CXXFLAGS) $(CXX14_FLAGS)

bin_PROGRAMS = hello
hello_SOURCES = main.cc
hello_LDADD = -L../modules/ -lnoop ../modules/libnoop.a
