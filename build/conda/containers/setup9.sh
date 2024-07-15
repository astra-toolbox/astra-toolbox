#!/bin/bash

set -e

. util.sh

setup_download $MINICONDA $CUDA102 $CUDA110

ctr=$(buildah from debian:9)

setup_fixup_debian9

setup_base
setup_conda
setup_cuda $CUDA102 $CUDA110
setup_cache_conda cudatoolkit 10.2 11.0


buildah commit $ctr astra-build-deb9

buildah rm $ctr
