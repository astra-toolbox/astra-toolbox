#!/bin/bash

set -e

. util.sh

setup_download $CUDA121

ctr=$(buildah from quay.io/pypa/manylinux2014_x86_64:latest)

setup_cuda $CUDA121

buildah run $ctr manylinux-interpreters ensure cp39-cp39 cp310-cp310 cp311-cp311 cp312-cp312
setup_pip_install python3.9 setuptools wheel numpy six Cython scipy
setup_pip_install python3.10 setuptools wheel numpy six Cython scipy
setup_pip_install python3.11 setuptools wheel numpy six Cython scipy
setup_pip_install python3.12 setuptools wheel numpy six Cython scipy

buildah commit $ctr astra-build-manylinux

buildah rm $ctr
