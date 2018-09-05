#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.68])
AC_INIT([nnfc], [0.1], [jemmons@cs.stanford.edu])
AM_INIT_AUTOMAKE([foreign])
AC_CONFIG_SRCDIR([src/nnfc/nnfc_CXXAPI.hh])
AC_CONFIG_HEADERS([config.h])

# include extra macros
AC_CONFIG_MACRO_DIRS([m4])
m4_include(m4/python.m4)
m4_include(m4/hdf5.m4)

# add third_party headers
TOP_SOURCE_DIR=`pwd`
THIRD_PARTY_CFLAGS="-isystem$TOP_SOURCE_DIR/third_party"
AC_SUBST([THIRD_PARTY_CFLAGS])

# Add compiler flags
CXX14_FLAGS="-std=c++17"
PICKY_CXXFLAGS="-Wall -Wextra -Weffc++ -Wno-non-virtual-dtor -Werror"
# OPTIMIZATION_FLAGS="-mtune=generic -g -O0 -fpic"
# OPTIMIZATION_FLAGS="-march=generic -mtune=generic -O2 -fpic"
OPTIMIZATION_FLAGS="-DNDEBUG -march=native -mtune=native -Ofast -fpic"
AC_SUBST([CXX14_FLAGS])
AC_SUBST([PICKY_CXXFLAGS])
AC_SUBST([OPTIMIZATION_FLAGS])

# add flags for python wrapper compilation
PYWRAPPER_EXTRA_CFLAGS="-std=c++17 -Wall -Wextra -Weffc++"
PYWRAPPER_EXTRA_LDFLAGS=""
# PYOPTIMIZATION_FLAGS="-march=generic -mtune=generic -g -O0 -fpic"
# PYOPTIMIZATION_FLAGS="-march=generic -mtune=generic -O2 -fpic"
PYOPTIMIZATION_FLAGS="-DNDEBUG -march=native -mtune=native -Ofast -fpic"
AC_SUBST([PYWRAPPER_EXTRA_CFLAGS])
AC_SUBST([PYWRAPPER_EXTRA_LDFLAGS])
AC_SUBST([PYOPTIMIZATION_FLAGS])

# Checks for programs.
AC_PROG_CXX
AM_PROG_LIBTOOL

# set the compiler version
AC_CHECK_PROG(GPP7_CHECK, g++-7, yes)
if test x"$GPP7_CHECK" != x"yes" ; then
    AC_MSG_ERROR([Please install g++-7 before installing.])
fi
AC_SUBST([CXX], [g++-7])

# Checks for assembler
AC_CHECK_PROGS([AS], [nasm], [none])
AS_IF([test x$AS == xnone], [
  AC_MSG_ERROR([nasm required for assembly. Install one of those to use SSE2.])
], [
  ASFLAGS="-DELF -D__x86_64__ -DPIC -f elf64"

AC_DEFINE([HAVE_SSE2], [1], [CPU supports SSE2])
  AC_DEFINE([ARCH_X86_64], [1], [64bit CPU])
])

AC_SUBST([AS])
AC_SUBST([ASFLAGS])

# Checks for libraries.
PKG_CHECK_MODULES([JPEG], [libturbojpeg])
AX_CHECK_HDF5_SERIAL()
PKG_CHECK_MODULES([AVFORMAT], [libavformat])
PKG_CHECK_MODULES([AVCODEC], [libavcodec])
PKG_CHECK_MODULES([AVUTIL], [libavutil])
PKG_CHECK_MODULES([SWSCALE], [libswscale])
PKG_CHECK_MODULES([FFTW3F], [fftw3f])

# Important: need to add flags for AVX, SSE, etc. to get good performance
AC_CHECK_FILE([/usr/include/eigen3/unsupported/Eigen/CXX11/Tensor], [
    EIGEN3_CFLAGS="-isystem /usr/include/eigen3/"
    EIGEN3_UNSUPPORTED_CFLAGS="-isystem /usr/include/eigen3/unsupported/"
    AC_SUBST([EIGEN3_CFLAGS])
    AC_SUBST([EIGEN3_UNSUPPORTED_CFLAGS])],
    [AC_MSG_ERROR([Missing eigen3 'Tensor' header. Something does not seem right...])
])

# check for python libraries
AC_PATH_PROG([PYTHON],[python3])
AX_PYTHON_VERSION([3.5.0])

AX_PYTHON_MODULE_VERSION([setuptools], [33.0.0], [python3])
AX_PYTHON_MODULE_VERSION([wheel], [0.29.0], [python3])
AX_PYTHON_MODULE_VERSION([h5py], [2.7.0], [python3])
AX_PYTHON_MODULE_VERSION([torch], [0.4.0], [python3])

AC_CONFIG_FILES([Makefile
         src/Makefile
         src/nn/Makefile
         src/codec/Makefile
         src/nnfc/Makefile
         src/frontend/Makefile
         src/frontend/simplenet9/Makefile
         src/frontend/arithcode/Makefile
         src/frontend/asmdct/Makefile
         python/Makefile
         tests/Makefile
         tests/arithcode/Makefile
])

AC_OUTPUT
