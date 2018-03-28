import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['mfc/src/mfc_wrapper.cc']
headers = ['mfc/src/mfc_wrapper.hh']
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['mfc/src/mfc_wrapper_cuda.cc']
    headers += ['mfc/src/mfc_wrapper_cuda.hh']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

include_dirs = map(os.path.abspath, ['../src/modules'])
library_dirs = map(os.path.abspath, ['../src/modules'])
runtime_library_dirs = map(os.path.abspath, ['../src/modules'])
libraries = ['noop']

ffi = create_extension(
    'mfc._ext.mfc_wrapper',
    package=True,
    headers=list(headers),
    sources=list(sources),
    include_dirs=list(include_dirs),
    library_dirs=list(library_dirs),
    runtime_library_dirs=list(runtime_library_dirs),
    libraries=list(libraries),
    define_macros=list(defines),
    relative_to=__file__,
    with_cuda=with_cuda
)

if __name__ == '__main__':
    ffi.build()
