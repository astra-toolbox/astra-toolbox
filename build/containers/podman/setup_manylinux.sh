#!/bin/bash

set -e

. util.sh

setup_download $CUDA129
setup_download $CUDA130

ctr=$(buildah from quay.io/pypa/manylinux2014_x86_64:latest)

setup_cuda $CUDA129
setup_cuda $CUDA130

buildah run $ctr manylinux-interpreters ensure cp39-cp39 cp310-cp310 cp311-cp311 cp312-cp312 cp313-cp313 cp313-cp313t cp314-cp314 cp314-cp314t
buildah run $ctr pipx upgrade auditwheel
setup_pip_install python3.9 setuptools wheel numpy six Cython scipy
setup_pip_install python3.10 setuptools wheel numpy six Cython scipy
setup_pip_install python3.11 setuptools wheel numpy six Cython scipy
setup_pip_install python3.12 setuptools wheel numpy six Cython scipy
setup_pip_install python3.13 setuptools wheel numpy six Cython scipy
setup_pip_install python3.13t setuptools wheel numpy six Cython scipy
setup_pip_install python3.14 setuptools wheel numpy six Cython scipy
setup_pip_install python3.14t setuptools wheel numpy six Cython scipy

buildah commit $ctr astra-build-manylinux

buildah rm $ctr
