import os
import sys
import re
import torch
import numpy

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension,  CppExtension, CUDAExtension

def get_def(header_filepath, definition_name):
    with open(header_filepath, 'r') as f:
        headerfile = f.read().strip()        
        version_found = re.search('\s*\#define\s*{def_name}\s*(\"(\d+\.)*\d+"?)[ \t]*$'.format(def_name=definition_name),
                                  headerfile, re.MULTILINE).group(1)
        return version_found
        
VERSION = get_def('../config.h', 'VERSION')
EXTENSION_NAME = 'nnfc._ext.nnfc_codec'
CUDA_AVAILABLE = torch.cuda.is_available()

'''
Defined the extension module below
'''
base_sources=['nnfc/src/nnfc_codec.cc',
              'nnfc/src/nnfc_encoder.cc', 'nnfc/src/nnfc_decoder.cc']
base_define_macros=[('_NNFC_VERSION', VERSION)]
base_include_dirs=[numpy.get_include(), '../src/modules', './extra_headers']
base_library_dirs=['../src/modules/.libs']
base_libraries=[]
base_extra_compile_args=[]
base_extra_link_args=['-lnnfc']


module = None
if CUDA_AVAILABLE:
    print('CUDA is available! Compiling the additional CUDA extension code.')

    module = CUDAExtension(name=EXTENSION_NAME,
                          sources=base_sources + ['nnfc/src/nnfc_cuda.cc'],
                          define_macros=base_define_macros,
                          include_dirs=base_include_dirs,
                          library_dirs=base_library_dirs,
                          libraries=base_libraries,
                          extra_compile_args=base_extra_compile_args + ['-D=_NNFC_CUDA_AVAILABLE=1'],
                          extra_link_args=base_extra_link_args
    )

else:
    module = CppExtension(name=EXTENSION_NAME,
                          sources=base_sources,
                          define_macros=base_define_macros + [],
                          include_dirs=base_include_dirs,
                          library_dirs=base_library_dirs,
                          libraries=base_libraries,
                          extra_compile_args=base_extra_compile_args,
                          extra_link_args=base_extra_link_args
    )

assert module is not None    
setup(
    name='nnfc_codec',
    version='0.0.0',
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
