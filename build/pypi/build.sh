#!/bin/sh

set -e

BRANCH=master
URL=https://github.com/astra-toolbox/astra-toolbox

echo "Cloning from ${URL}"
echo "        branch: ${BRANCH}"

cd /root
git clone --depth 1 --branch ${BRANCH} ${URL}

cd astra-toolbox/build/linux

./autogen.sh
for PYTHON_VERSION in 3.9 3.10 3.11 3.12 3.13; do
    ./configure --with-python=python${PYTHON_VERSION} \
                --with-cuda=/usr/local/cuda-12.5 \
                --with-install-type=module
    make -j 20 ASTRA_CONFIG=pypi_linux_cuda libastra.la py
    auditwheel repair --plat manylinux2014_x86_64 --exclude "libcudart.so.*" --exclude "libcufft.so.*" python/dist/*.whl
    mv wheelhouse/*.whl /out
    make clean
done
