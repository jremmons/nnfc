AM_CPPFLAGS = $(CXX14_FLAGS) $(THIRD_PARTY_CFLAGS) \
              $(JPEG_CFLAGS) \
              $(EIGEN3_CFLAGS) $(EIGEN3_UNSUPPORTED_CFLAGS) \
              -I$(srcdir)/..

AM_CXXFLAGS = $(PICKY_CXXFLAGS) $(OPTIMIZATION_FLAGS)

SUFFIXES = .asm
.asm.o:
	$(AS) $(ASFLAGS) $<

noinst_LIBRARIES = libcodec.a

libcodec_a_SOURCES = arithmetic_coder.hh arithmetic_coder_common.hh arithmetic_probability_model.hh \
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
