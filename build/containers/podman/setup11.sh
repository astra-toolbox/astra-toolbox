#!/bin/bash

set -e

. util.sh

setup_download $MINICONDA $CUDA111 $CUDA112 $CUDA113 $CUDA114 $CUDA115 $CUDA116 $CUDA117 $CUDA118 $CUDA120 $CUDA121 $CUDA122

ctr=$(buildah from debian:11)

setup_base
setup_conda
setup_cuda $CUDA111 $CUDA112 $CUDA113 $CUDA114 $CUDA115 $CUDA116 $CUDA117 $CUDA118
setup_cache_conda cuda-cudart 11.6 11.7 11.8
setup_cache_conda libcufft 10.7.0 10.7.1 10.7.2 10.9.0
setup_cache_conda cudatoolkit 11.3 11.4 11.5 11.6 11.7 11.8
buildah run $ctr conda create -y -n prep -c nvidia --download-only "cudatoolkit=11.1.*,<11.1.74"
buildah run $ctr conda create -y -n prep -c nvidia --download-only "cudatoolkit=11.2.*,<11.2.72"

buildah commit $ctr astra-build-deb11

buildah rm $ctr
