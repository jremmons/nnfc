#!/bin/bash

mkdir -p coco/images
cd coco/images
wget -c https://pjreddie.com/media/files/val2014.zip
unzip -q val2014.zip

cd ..
wget -c https://pjreddie.com/media/files/coco/labels.tgz
tar xzf labels.tgz
