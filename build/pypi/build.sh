#!/bin/sh

set -e

BRANCH=master
URL=https://github.com/astra-toolbox/astra-toolbox
CUDA_RPATHS='$ORIGIN/../nvidia/cuda_runtime/lib:$ORIGIN/../nvidia/cufft/lib'

echo "Cloning from ${URL}"
echo "        branch: ${BRANCH}"

cd /root
git clone --depth 1 --branch ${BRANCH} ${URL}

cd astra-toolbox/build/linux

./autogen.sh
for PYTHON_VERSION in 3.9 3.10 3.11 3.12 3.13; do
    ./configure --with-python=python${PYTHON_VERSION} \
                --with-cuda=/usr/local/cuda-12.1 \
                --with-install-type=module \
                --with-python-wheel-platform=manylinux2014_x86_64
    make -j 20 libastra.la
    patchelf --set-rpath $CUDA_RPATHS .libs/$(source libastra.la; echo $dlname)
    make python-wheel
    cp python/dist/*.whl /out
    make clean
done
