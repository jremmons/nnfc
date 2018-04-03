#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $DIR
python setup.py clean --all
python setup.py bdist_wheel
pip install --force-reinstall --upgrade dist/*.whl
