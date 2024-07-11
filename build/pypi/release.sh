#!/bin/bash

set -e

D=`mktemp -d`
cp build.sh $D

podman run --rm -v $D:/out:z astra-build-manylinux /bin/bash /out/build.sh

rm -f $D/build.sh

mkdir -p pkgs
mv $D/* pkgs
rmdir $D

