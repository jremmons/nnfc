#!/bin/bash

lib_count=$(ls ../python/build/ | grep 'lib' | wc -l)
if [[ $lib_count != '1' ]]; then
    echo -e '\n\n'
    echo "ERROR: too many python lib dirs! (count: $lib_count) (ls: $(ls ../python/build/))"
    echo -e '\n\n'
    exit -1
fi

libdir=../python/build/$(ls ../python/build/ | grep 'lib')

export PYTHONPATH=$libdir:$PYTHONPATH
exec "$@"
