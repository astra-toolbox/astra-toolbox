#!/bin/bash

set -e

D=`mktemp -d`
cp build.sh $D

V=2.5.99
B=0

podman run --rm -v $D:/out:z astra-build-cuda11 /bin/bash /out/build.sh $V $B cuda11 full
podman run --rm -v $D:/out:z astra-build-cuda12 /bin/bash /out/build.sh $V $B cuda12
podman run --rm -v $D:/out:z astra-build-cuda13 /bin/bash /out/build.sh $V $B cuda13 full

rm -f $D/build.sh

mkdir -p pkgs
mv $D/* pkgs
rmdir $D

