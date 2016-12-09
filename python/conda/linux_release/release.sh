#!/bin/bash

D=`mktemp -d`

[ -f buildenv/cuda_5.5.22_linux_64.run ] || (cd buildenv; wget http://developer.download.nvidia.com/compute/cuda/5_5/rel/installers/cuda_5.5.22_linux_64.run )
[ -f buildenv/Miniconda3-4.2.12-Linux-x86_64.sh ] || (cd buildenv; wget https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh )

docker build -t astra-build-env buildenv
#docker build --no-cache --build-arg=BUILD_NUMBER=0 -t astra-builder builder
docker build --no-cache -t astra-builder builder

docker run --name astra-build-cnt -v $D:/out:z astra-builder /bin/bash -c "cp /root/miniconda3/conda-bld/linux-64/*astra* /out"

mkdir -p pkgs
mv $D/* pkgs
rmdir $D

docker rm astra-build-cnt
docker rmi astra-builder

