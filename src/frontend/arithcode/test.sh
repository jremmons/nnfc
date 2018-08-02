#!/bin/sh

./arith_encode example-text/text.txt compressed.bin

./arith_decode compressed.bin output.txt

diff -q example-text/text.txt output.txt
if [ $? != 0 ]; then
    echo "failure!"
else
    echo "success!"
fi
