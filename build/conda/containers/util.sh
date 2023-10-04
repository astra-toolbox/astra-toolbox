#!/bin/bash

set -e

MINICONDA=https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
CUDA102=https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
CUDA110=https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run
CUDA111=https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
CUDA112=https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_460.32.03_linux.run
CUDA113=https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run
#CUDA114=https://developer.download.nvidia.com/compute/cuda/11.4.1/local_installers/cuda_11.4.1_470.57.02_linux.run
CUDA114=https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda_11.4.4_470.82.01_linux.run
#CUDA115=https://developer.download.nvidia.com/compute/cuda/11.5.1/local_installers/cuda_11.5.1_495.29.05_linux.run
CUDA115=https://developer.download.nvidia.com/compute/cuda/11.5.2/local_installers/cuda_11.5.2_495.29.05_linux.run
#CUDA116=https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
CUDA116=https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run
#CUDA117=https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
CUDA117=https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
CUDA118=https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
CUDA120=https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda_12.0.1_525.85.12_linux.run
CUDA121=https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
CUDA122=https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run

setup_download() {
  mkdir -p sw
  chcon unconfined_u:object_r:container_file_t:s0 sw || /bin/true
  for F in $@; do
    [ -f sw/`basename $F` ] || (cd sw; wget $F)
    chcon unconfined_u:object_r:container_file_t:s0 sw/`basename $F` || /bin/true
  done
}

setup_fixup_debian9() {
  buildah run $ctr sed -i 's|deb.debian.org|archive.debian.org|g' /etc/apt/sources.list
  buildah run $ctr sed -i 's|security.debian.org|archive.debian.org|g' /etc/apt/sources.list
  buildah run $ctr sed -i '/stretch-updates/d' /etc/apt/sources.list
}


setup_base() {
  echo Setting up $ctr
  buildah config --env DEBIAN_FRONTEND=noninteractive $ctr
  buildah run $ctr apt-get update
  buildah run $ctr apt-get install -y perl-modules build-essential autoconf libtool automake libboost-dev git libxml2
  buildah run $ctr apt-get install -y git-lfs || /bin/true
}

setup_conda() {
  echo Installing $(basename $MINICONDA)
  buildah run --volume `pwd`/sw:/sw:ro,z $ctr bash /sw/$(basename $MINICONDA) -b
  buildah config --env PATH=/root/miniconda3/bin:$(buildah run $ctr printenv PATH) $ctr
  buildah run $ctr conda install -y conda-build conda-verify
}

setup_cuda() {
  for C in $@; do
    echo Installing $(basename $C)
    buildah run --volume `pwd`/sw:/sw:ro,z $ctr bash /sw/$(basename $C) --toolkit --silent
  done
  buildah run $ctr rm -f /usr/local/cuda
}

setup_cache_conda() {
  N=$1
  shift
  for C in $@; do
    buildah run $ctr conda create -y -n prep -c nvidia --download-only $N=$C
  done
}

