AM_CPPFLAGS = $(CXX14_FLAGS) $(JPEG_CFLAGS) \
              $(EIGEN3_CFLAGS) $(EIGEN3_UNSUPPORTED_CFLAGS) \
              -I$(srcdir)/..

AM_CXXFLAGS = $(PICKY_CXXFLAGS) -fPIC 

SUFFIXES = .asm
.asm.o:
	$(AS) $(ASFLAGS) -I$(srcdir)/../asm/ $<

noinst_LIBRARIES = libcodec.a

libcodec_a_SOURCES = arithmetic_encoder.hh \
                     fastdct.hh fastdct.cc \
                     jpeg.hh jpeg.cc \
                     mpeg.hh mpeg.cc \
                     swizzle.hh swizzle.cc \
                     utils.hh utils.cc \
                     tjdct/jfdctint-sse2.asm \ 
                     tjdct/jidctint-sse2.asm \
                     tjdct/jfdctfst-sse2.asm \
                     tjdct/jidctfst-sse2.asm \
                     tjdct/jquanti-sse2.asm
