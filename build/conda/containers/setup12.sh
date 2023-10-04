#!/bin/bash

set -e

. util.sh

setup_download $MINICONDA $CUDA120 $CUDA121 $CUDA122

ctr=$(buildah from debian:12)

setup_base
setup_conda
setup_cuda $CUDA120 $CUDA121 $CUDA122
setup_cache_conda cuda-cudart 12.0 12.1 12.2
setup_cache_conda libcufft 11.0.0 11.0.1 11.0.2 11.0.8

buildah commit $ctr astra-build-deb12

buildah rm $ctr
