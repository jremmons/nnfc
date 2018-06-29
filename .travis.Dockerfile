FROM ubuntu:18.04

MAINTAINER sadjad "https://github.com/sadjad"

RUN apt-get update -qq
RUN apt-get install -q -y libturbojpeg0-dev gcc-7 g++-7 nasm libhdf5-dev \
                          hdf5-helpers libeigen3-dev python3-dev python-dev \
                          libavcodec-dev libavformat-dev libavutil-dev \
                          libfftw3-dev python3-pip git
RUN apt-get install -q -y automake libtool pkg-config
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 99
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 99

RUN pip3 install --quiet --upgrade pip
RUN pip3 install --quiet --upgrade setuptools
RUN pip3 install --quiet --upgrade wheel
RUN pip3 install --quiet --upgrade numpy
RUN pip3 install --quiet --upgrade torch
RUN pip3 install --quiet --upgrade torchvision
RUN pip3 install --quiet --upgrade h5py

RUN useradd --create-home --shell /bin/bash user
COPY . /home/user/nnfc/
RUN chown user -R /home/user/nnfc/

ENV LANG C.UTF-8
ENV LANGUAGE C:en
ENV LC_ALL C.UTF-8

USER user
