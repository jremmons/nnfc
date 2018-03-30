#!/bin/sh

./autogen.sh
./configure.sh
make -j

cd python
python setup.py clean --all
python setup.py install
cd ..
