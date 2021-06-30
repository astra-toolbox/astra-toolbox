#!/bin/bash

set -e

D=`mktemp -d`

for F in https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_460.32.03_linux.run ; do
	[ -f buildenv/`basename $F` ] || (cd buildenv; wget $F )
done

for F in https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run https://developer.nvidia.com/compute/cuda/8.0/Prod2/patches/2/cuda_8.0.61.2_linux-run https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda_9.2.148_396.37_linux https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run ; do
	[ -f buildenv/`basename $F` ] || (cd buildenv; wget $F )
done



docker build -t astra-build-env buildenv
docker build -t astra-build-env-archive -f buildenv/archive.Dockerfile buildenv

cp buildenv/build.sh $D

V=1.9.9.dev7

docker run -v $D:/out:z astra-build-env-archive /bin/bash /out/build.sh $V 0 archive
docker run -v $D:/out:z astra-build-env /bin/bash /out/build.sh $V 0

rm -f $D/build.sh

mkdir -p pkgs
mv $D/* pkgs
rmdir $D

