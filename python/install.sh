#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $DIR
python3 setup.py clean --all
python3 setup.py --plat-tag linux_x86-64 bdist_wheel
pip3 install --force-reinstall --upgrade dist/*.whl
