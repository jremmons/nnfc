AM_CPPFLAGS = -I$(srcdir)/../../codec 

AM_CXXFLAGS = $(PICKY_CXXFLAGS) $(CXX14_FLAGS) \
              -L$(srcdir)/../../codec/ -lcodec 

bin_PROGRAMS = arith_encode arith_decode

arith_encode_SOURCES = encode.cc 
arith_encode_LDADD = 

arith_decode_SOURCES = decode.cc 
arith_decode_LDADD = 
