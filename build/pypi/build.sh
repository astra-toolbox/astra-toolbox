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

./configure --with-python=python3.9 --with-cuda=/usr/local/cuda-12.1 --with-install-type=module
make -j 20 ASTRA_CONFIG=pypi_linux_cuda all
auditwheel repair --plat manylinux2014_x86_64 --exclude "libcudart.so.*" --exclude "libcufft.so.*" python/dist/*.whl
mv wheelhouse/*.whl /out
make clean

./configure --with-python=python3.10 --with-cuda=/usr/local/cuda-12.1 --with-install-type=module
make -j 20 ASTRA_CONFIG=pypi_linux_cuda all
auditwheel repair --plat manylinux2014_x86_64 --exclude "libcudart.so.*" --exclude "libcufft.so.*" python/dist/*.whl
mv wheelhouse/*.whl /out
make clean

./configure --with-python=python3.11 --with-cuda=/usr/local/cuda-12.1 --with-install-type=module
make -j 20 ASTRA_CONFIG=pypi_linux_cuda all
auditwheel repair --plat manylinux2014_x86_64 --exclude "libcudart.so.*" --exclude "libcufft.so.*" python/dist/*.whl
mv wheelhouse/*.whl /out
make clean

./configure --with-python=python3.12 --with-cuda=/usr/local/cuda-12.1 --with-install-type=module
make -j 20 ASTRA_CONFIG=pypi_linux_cuda all
auditwheel repair --plat manylinux2014_x86_64 --exclude "libcudart.so.*" --exclude "libcufft.so.*" python/dist/*.whl
mv wheelhouse/*.whl /out
make clean

./configure --with-python=python3.13 --with-cuda=/usr/local/cuda-12.1 --with-install-type=module
make  -j 20 ASTRA_CONFIG=pypi_linux_cuda all
auditwheel repair --plat manylinux2014_x86_64 --exclude "libcudart.so.*" --exclude "libcufft.so.*" python/dist/*.whl
mv wheelhouse/*.whl /out
make clean
