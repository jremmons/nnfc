# NNFC: the neural network (machine) feature codec!

[![Build Status](https://travis-ci.com/jremmons/machine-feature-codec.svg?token=KdZxgGzcW16Ms5dBx8kN&branch=master)](https://travis-ci.com/jremmons/machine-feature-codec)

**Description**: a video and image compression method for machine learned features.

**Travis-CI link**: [https://travis-ci.com/jremmons/machine-feature-codec](https://travis-ci.com/jremmons/machine-feature-codec)

**Paper references**: [https://github.com/jremmons/machine-feature-codec-papers](https://github.com/jremmons/machine-feature-codec-papers)

## Build from source instructions

### Prerequistes 

```bash
sudo apt-get install build-essential
# sudo apt-get install ... add more build prereqs
```

### Build

```bash
# Clone the repo
git clone https://github.com/jremmons/machine-feature-codec.git
cd machine-feature-codec

# Build the library
./autogen.sh
./configure
make -j 

# Build the PyTorch wrapper
python python/setup.py bdist_wheel

# Install the PyTorch wrapper
pip install dist/nnff-*
```
