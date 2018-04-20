import os
import torch
from torch.utils.ffi import create_extension

this_dir = os.path.abspath(os.path.dirname(__file__))

sources = ['nnfc/src/nnfc_wrapper.cc', 'nnfc/src/noop_wrapper.cc']
headers = ['nnfc/src/nnfc_wrapper.hh', 'nnfc/src/noop_wrapper.hh']
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['nnfc/src/cuda_functions.cc']
    headers += ['nnfc/src/cuda_functions.hh']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

include_dirs = map(lambda x: os.path.join(this_dir, x), ['../src/modules'])
library_dirs = map(lambda x: os.path.join(this_dir, x), ['../src/modules', '../src/modules/.libs'])
runtime_library_dirs = map(lambda x: os.path.join(this_dir, x), ['../src/modules'])
libraries = []

extra_compile_args = os.environ.get("CXXFLAGS", "").split() + []
extra_link_args = ['-Wl,-Bstatic', '-lnoop',
                   '-Wl,-Bstatic', '-lnnfc',
                   '-Wl,-Bdynamic', '-lturbojpeg',
                   '-Wl,-Bdynamic', '-lpthread']

extra_objects = ['../src/modules/.libs/libnoop.a', '../src/modules/.libs/libnnfc.a']

# TODO(jremmons) provide option for dynamic linking external libraries
# Note: we currently statically link so that our experiments will always use identical libraries

ffi = create_extension(
    'nnfc._ext.nnfc_wrapper',
    package=True,
    headers=list(headers),
    sources=list(sources),
    include_dirs=list(include_dirs),
    library_dirs=list(library_dirs),
    runtime_library_dirs=list(runtime_library_dirs),
    libraries=list(libraries),
    define_macros=list(defines),
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    extra_objects=extra_objects,
    relative_to=__file__,
    with_cuda=with_cuda
)

if __name__ == '__main__':
    ffi.build()
