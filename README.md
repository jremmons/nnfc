# NNFC: the neural network (machine) feature codec!

[![Build Status](https://travis-ci.org/jremmons/nnfc.svg?branch=master)](https://travis-ci.org/jremmons/nnfc)

**Description**: a video and image compression method for machine learned features.

**Travis-CI link**: [https://travis-ci.com/jremmons/nnfc](https://travis-ci.com/jremmons/nnfc)

**Paper references**: [https://github.com/jremmons/nnfc-papers](https://github.com/jremmons/nnfc-papers)

## Build from source instructions

### Prerequistes 

```bash
sudo apt-get install build-essential
# sudo apt-get install ... add more build prereqs
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
