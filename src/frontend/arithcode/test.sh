#!/bin/sh

./arith_encode example-text/text.small.txt compressed.bin

rm -f output.txt
./arith_decode compressed.bin output.txt

diff -q example-text/text.small.txt output.txt
if [ $? != 0 ]; then
    echo "failure!"
else
    echo "success!"
fi

# cat output.txt
# echo
# cat example-text/text.txt
