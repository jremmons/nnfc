import os
import sys
import re
import torch
import numpy

from setuptools import setup, find_packages, Extension 
from torch.utils.cpp_extension import BuildExtension, include_paths, library_paths

def get_def(header_filepath, definition_name):
    with open(header_filepath, 'r') as f:
        headerfile = f.read().strip()        
        version_found = re.search('\s*\#define\s*{def_name}\s*\"((\d+\.)*\d+?)"[ \t]*$'.format(def_name=definition_name),
                                  headerfile, re.MULTILINE).group(1)
        return version_found
        
VERSION = get_def('../config.h', 'VERSION')
EXTENSION_NAME = 'nnfc._ext.nnfc_codec'
CUDA_AVAILABLE = torch.cuda.is_available()

pytorch_include = []
for lib in include_paths(cuda=CUDA_AVAILABLE):
    pytorch_include += ['-isystem', lib]

pytorch_libdirs = library_paths(cuda=CUDA_AVAILABLE)
pytorch_libs = ['cudart'] if CUDA_AVAILABLE else [] 
pytorch_defines = [('_NNFC_CUDA_AVAILABLE', 1)] if CUDA_AVAILABLE else []

module = Extension(EXTENSION_NAME,
                   sources=['nnfc/src/nnfc_codec.cc', 'nnfc/src/nnfc_cuda.cc', 
                            'nnfc/src/nnfc_encoder.cc', 'nnfc/src/nnfc_decoder.cc'],
                   define_macros=[('_NNFC_VERSION', '"'+VERSION+'"')] + pytorch_defines,
                   include_dirs=[numpy.get_include()],
                   library_dirs=['../src/nnfc/.libs'] + pytorch_libdirs,
                   libraries=[] + pytorch_libs,
                   extra_compile_args=['-I../src/nn', '-I../src/nnfc',
                                       '-isystem', './extra_headers'] + pytorch_include,
                   extra_link_args=['-Wl,-Bdynamic', '-lnnfc']
                   )

setup(
    name=EXTENSION_NAME,
    version=VERSION,
    description='',
    long_description='',
    url='',
    author='John R. Emmons',
    author_email='jemmons@cs.stanford.edu',
    install_requires=[],
    setup_requires=[],
    packages=find_packages(exclude=['build', 'extra_headers']),
    ext_modules=[module],
    cmdclass={
        'build_ext': BuildExtension
    }
)
