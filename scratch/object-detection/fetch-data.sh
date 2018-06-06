#!/bin/bash

mkdir -p coco/images
cd coco/images
wget -c https://s3-us-west-2.amazonaws.com/nnfc-data/coco/val2014.zip
unzip -q val2014.zip

cd ..
wget -c https://s3-us-west-2.amazonaws.com/nnfc-data/coco/labels.tgz
tar xzf labels.tgz
