# NNFC: the neural network feature codec!

[![Build Status](https://travis-ci.org/jremmons/nnfc.svg?branch=master)](https://travis-ci.org/jremmons/nnfc)

**Description**: a video and image compression method for machine learned features.

**Travis-CI link**: [https://travis-ci.com/jremmons/nnfc](https://travis-ci.com/jremmons/nnfc)

**Paper references**: [https://github.com/jremmons/nnfc-papers](https://github.com/jremmons/nnfc-papers)

## Build from source instructions

### Prerequistes 

```bash
sudo add-apt-repository -y ppa:keithw/libjpeg-turbo-backports  

sudo apt-get update

sudo apt-get install build-essential
sudo apt-get install python3-dev
sudo apt-get install libturbojpeg0-dev 
```

### Build

```bash
# Clone the repo
git clone https://github.com/jremmons/nnfc.git
cd nnfc

# Build the library
./autogen.sh
./configure
make -j 

# Build the PyTorch wrapper
python python/setup.py bdist_wheel

# Install the PyTorch wrapper
pip install dist/nnff-*
```
