#!/bin/bash

set -e

D=`mktemp -d`
cp build.sh $D

V=2.1.3
B=0

podman run --rm -v $D:/out:z astra-build-deb9 /bin/bash /out/build.sh $V $B deb9 full
podman run --rm -v $D:/out:z astra-build-deb11 /bin/bash /out/build.sh $V $B deb11 full
podman run --rm -v $D:/out:z astra-build-deb12 /bin/bash /out/build.sh $V $B deb12

rm -f $D/build.sh

mkdir -p pkgs
mv $D/* pkgs
rmdir $D

