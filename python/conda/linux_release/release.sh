#!/bin/bash

set -e

D=`mktemp -d`

for F in https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run https://developer.nvidia.com/compute/cuda/8.0/Prod2/patches/2/cuda_8.0.61.2_linux-run https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda_9.2.148_396.37_linux https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.168_418.67_linux.run http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run ; do
	[ -f buildenv/`basename $F` ] || (cd buildenv; wget $F )
done

docker build -t astra-build-env buildenv

cp buildenv/build.sh $D

docker run -v $D:/out:z astra-build-env /bin/bash /out/build.sh 1.9.9.dev4 0

rm -f $D/build.sh

mkdir -p pkgs
mv $D/* pkgs
rmdir $D

