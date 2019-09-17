#!/bin/sh

case `uname` in
  Darwin*)
    CUDA_ROOT=/usr/local/cuda
    CC=gcc
    CXX=g++
    ;;
  Linux*)
    [ -n "$cudatoolkit" ] || exit 1
    CUDA_ROOT=/usr/local/cuda-$cudatoolkit
    ;;
esac

[ -x "$CUDA_ROOT" ] || exit 1

cd $SRC_DIR/build/linux

$SRC_DIR/build/linux/autogen.sh

# Add C++11 to compiler flags, mostly to work around a boost bug.
# Since we require cudatoolkit>=8.0 nvcc supports this.
NVCC=$CUDA_ROOT/bin/nvcc
EXTRA_NVCCFLAGS="--std=c++11"


$SRC_DIR/build/linux/configure --with-install-type=prefix --with-cuda=$CUDA_ROOT --prefix=$PREFIX NVCCFLAGS="-ccbin $CC -I$PREFIX/include $EXTRA_NVCCFLAGS" CC=$CC CXX=$CXX CPPFLAGS="-I$PREFIX/include"

# Clean, because we may be re-using this source tree when building
# multiple variants of this conda package.
make clean

make -j $CPU_COUNT
make -j $CPU_COUNT install-dev


test -d $CUDA_ROOT/lib64 && LIBPATH="$CUDA_ROOT/lib64" || LIBPATH="$CUDA_ROOT/lib"

case `uname` in
  Darwin*)
    cp -P $LIBPATH/libcudart.*.dylib $PREFIX/lib
    cp -P $LIBPATH/libcufft.*.dylib $PREFIX/lib
    ;;
esac
