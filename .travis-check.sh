#!/bin/bash -ex

cd /home/user/nnfc

./autogen.sh
./configure
make -j4
make check || (cat tests/test-suite.log tests/pythonpath.txt && exit 1)
