#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(readlink -f $DIR/../src/nnfc/.libs)

lib_count=$(ls $DIR/../python/build/ | grep 'lib' | wc -l)
if [[ $lib_count != '1' ]]; then
    echo -e '\n\n'
    echo "ERROR: too many python lib dirs! (count: $lib_count) (ls: $(ls ../python/build/))"
    echo -e '\n\n'
    exit -1
fi

libdir=../python/build/$(ls $DIR/../python/build/ | grep 'lib')

export PYTHONPATH=$(readlink -f $libdir):$PYTHONPATH

export NNFC_TEST_TMPDIR=/tmp/nnfc_test_tmpdir
mkdir -p $NNFC_TEST_TMPDIR

exec "$@"
