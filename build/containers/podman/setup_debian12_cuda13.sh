#!/bin/bash

set -e

. util.sh

setup_download $MINICONDA $CUDA130 $CUDA131 $CUDA132 $CUDA133

ctr=$(buildah from debian:12)

setup_base
setup_conda
setup_cuda $CUDA130 $CUDA131 $CUDA132 $CUDA133
setup_cache_conda cuda-cudart 13.0 13.1 13.2 13.3
setup_cache_conda libcufft 12.0.0 12.1.0 12.2.0 12.3.0

buildah commit $ctr astra-build-cuda13

buildah rm $ctr
