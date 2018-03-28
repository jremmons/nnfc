import os
import sys

from setuptools import setup, find_packages
import build

this_file = os.path.dirname(__file__)

setup(
    name='mfc',
    version='0.1',
    description='',
    long_description='',
    url='',
    author='John R. Emmons',
    author_email='jemmons@cs.stanford.edu',
    install_requires=['cffi>=1.0.0'],
    setup_requires=['cffi>=1.0.0'],
    packages=find_packages(exclude=['build']),
    ext_package='',
    cffi_modules=[
        os.path.join(this_file, 'build.py:ffi')
    ],
)
