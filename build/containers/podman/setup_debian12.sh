#!/bin/bash

set -e

. util.sh

setup_download $MINICONDA $CUDA120 $CUDA121 $CUDA122 $CUDA123 $CUDA124 $CUDA125 $CUDA126 $CUDA128

ctr=$(buildah from debian:12)

setup_base
setup_conda
setup_cuda $CUDA120 $CUDA121 $CUDA122 $CUDA123 $CUDA124 $CUDA125 $CUDA126 $CUDA128
setup_cache_conda cuda-cudart 12.0 12.1 12.2 12.3 12.4 12.5 12.6 12.8
setup_cache_conda libcufft 11.0.8 11.0.12 11.2.1 11.2.3 11.3.0 11.3.3

buildah commit $ctr astra-build-deb12

buildah rm $ctr
