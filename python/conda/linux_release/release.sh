#!/bin/bash

D=`mktemp -d`

for F in https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh  http://developer.download.nvidia.com/compute/cuda/5_5/rel/installers/cuda_5.5.22_linux_64.run http://developer.download.nvidia.com/compute/cuda/6_0/rel/installers/cuda_6.0.37_linux_64.run  http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/cuda_7.0.28_linux.run http://developer.download.nvidia.com/compute/cuda/7_0/Prod/cufft_update/cufft_patch_linux.tar.gz http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.18_linux.run https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run https://developer.nvidia.com/compute/cuda/8.0/Prod2/patches/2/cuda_8.0.61.2_linux-run; do
	[ -f buildenv/`basename $F` ] || (cd buildenv; wget $F )
done

docker build -t astra-build-env buildenv

cp buildenv/build.sh $D

docker run -v $D:/out:z astra-build-env /bin/bash /out/build.sh

rm -f $D/build.sh

mkdir -p pkgs
mv $D/* pkgs
rmdir $D

