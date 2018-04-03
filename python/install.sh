#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python setup.py clean --all
python setup.py bdist_wheel
pip install --force-reinstall --upgrade $DIR/dist/*.whl
