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
#m4_include(m4/boost.m4)

# Add picky CXXFLAGS
CXX14_FLAGS="-std=c++17"
PICKY_CXXFLAGS="-Wall -Wextra -Weffc++ -Wno-non-virtual-dtor -Werror"
AC_SUBST([CXX14_FLAGS])
AC_SUBST([PICKY_CXXFLAGS])

# add flags for python wrapper compilation
PYWRAPPER_EXTRA_CFLAGS="-std=c++17 -Wall -Wextra -Weffc++"
PYWRAPPER_EXTRA_LDFLAGS=""
AC_SUBST([PYWRAPPER_EXTRA_CFLAGS])
AC_SUBST([PYWRAPPER_EXTRA_LDFLAGS])

# Checks for programs.
AC_PROG_CXX
AM_PROG_LIBTOOL

# Checks for libraries.
PKG_CHECK_MODULES([JPEG], [libturbojpeg])
AX_CHECK_HDF5_SERIAL()

PKG_CHECK_MODULES([AVFORMAT], [libavformat])
PKG_CHECK_MODULES([AVCODEC], [libavcodec])
PKG_CHECK_MODULES([AVUTIL], [libavutil])
PKG_CHECK_MODULES([SWSCALE], [libswscale])

# Important: need to add flags for AVX, SSE, etc. to get good performance
AC_CHECK_FILE([/usr/include/eigen3/unsupported/Eigen/CXX11/Tensor], [
    EIGEN3_CFLAGS="-isystem /usr/include/eigen3/"
    EIGEN3_UNSUPPORTED_CFLAGS="-isystem /usr/include/eigen3/unsupported/"
    AC_SUBST([EIGEN3_CFLAGS])
    AC_SUBST([EIGEN3_UNSUPPORTED_CFLAGS])],
    [AC_MSG_ERROR([Missing eigen3 'Tensor' header. Something does not seem right...])
])

# Checks for header files.

# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.

# define macros for checking python libraries

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
         python/Makefile
         tests/Makefile
])

AC_OUTPUT
