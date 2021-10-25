#!/bin/bash

set -e

D=`mktemp -d`

for F in \
  https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh \
 https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run \
 https://developer.nvidia.com/compute/cuda/8.0/Prod2/patches/2/cuda_8.0.61.2_linux-run \
 https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run \
 https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda_9.2.148_396.37_linux \
 https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux \
 http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run \
 http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run \
; do
	[ -f buildenv.deb8/`basename $F` ] || (cd buildenv.deb8; wget $F )
done
for F in \
  https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh \
 https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run \
 https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run \
; do
	[ -f buildenv.deb9/`basename $F` ] || (cd buildenv.deb9; wget $F )
done

for F in \
  https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh \
 https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_460.32.03_linux.run \
 https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run \
 https://developer.download.nvidia.com/compute/cuda/11.4.1/local_installers/cuda_11.4.1_470.57.02_linux.run \
 https://developer.download.nvidia.com/compute/cuda/11.5.0/local_installers/cuda_11.5.0_495.29.05_linux.run \
; do
	[ -f buildenv.deb11/`basename $F` ] || (cd buildenv.deb11; wget $F )
done


docker build -t astra-build-env-deb8 buildenv.deb8
docker build -t astra-build-env-deb9 buildenv.deb9
docker build -t astra-build-env-deb11 buildenv.deb11

cp build.sh $D

V=2.0.0

docker run -v $D:/out:z astra-build-env-deb8 /bin/bash /out/build.sh $V 0 deb8
docker run -v $D:/out:z astra-build-env-deb9 /bin/bash /out/build.sh $V 0 deb9 full
#Disable this until cython is available for python 3.10 in conda
#docker run -v $D:/out:z astra-build-env-deb11 /bin/bash /out/build.sh $V 0 deb11 full
docker run -v $D:/out:z astra-build-env-deb11 /bin/bash /out/build.sh $V 0 deb11

rm -f $D/build.sh

mkdir -p pkgs
mv $D/* pkgs
rmdir $D

