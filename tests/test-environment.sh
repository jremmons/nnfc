#!/bin/bash

# we are statically linking libnnfc for now so we don't need to set
# the dynamic linker path. If that changes, uncommment the line below.
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../src/nnfc/.libs

lib_count=$(ls ../python/build/ | grep 'lib' | wc -l)
if [[ $lib_count != '1' ]]; then
    echo -e '\n\n'
    echo "ERROR: too many python lib dirs! (count: $lib_count) (ls: $(ls ../python/build/))"
    echo -e '\n\n'
    exit -1
fi

libdir=../python/build/$(ls ../python/build/ | grep 'lib')

export PYTHONPATH=$libdir:$PYTHONPATH

export NNFC_TEST_TMPDIR=/tmp/nnfc_test_tmpdir
mkdir -p $NNFC_TEST_TMPDIR

exec "$@"
