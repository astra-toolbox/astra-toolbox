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

# Add C++11 to compiler flags if nvcc supports it, mostly to work around a boost bug
NVCC=$CUDA_ROOT/bin/nvcc
echo "int main(){return 0;}" > $CONDA_PREFIX/test.cu
$NVCC $CONDA_PREFIX/test.cu -ccbin $CC --std=c++11 -o $CONDA_PREFIX/test.out > /dev/null 2>&1 && EXTRA_NVCCFLAGS="--std=c++11" || true
rm -f $CONDA_PREFIX/test.out

$SRC_DIR/build/linux/configure --with-install-type=prefix --with-cuda=$CUDA_ROOT --prefix=$CONDA_PREFIX NVCCFLAGS="-ccbin $CC -I$CONDA_PREFIX/include $EXTRA_NVCCFLAGS" CC=$CC CXX=$CXX CPPFLAGS="-I$CONDA_PREFIX/include"

# Clean, because we may be re-using this source tree when building
# multiple variants of this conda package.
make clean

make -j $CPU_COUNT install-libraries


test -d $CUDA_ROOT/lib64 && LIBPATH="$CUDA_ROOT/lib64" || LIBPATH="$CUDA_ROOT/lib"

case `uname` in
  Darwin*)
    cp -P $LIBPATH/libcudart.*.dylib $CONDA_PREFIX/lib
    cp -P $LIBPATH/libcufft.*.dylib $CONDA_PREFIX/lib
    ;;
  Linux*)
    if [ "$cudatoolkit" = "7.0" ]; then
      # For some reason conda-build adds these symlinks automatically for
      # cudatoolkit-5.5 and 6.0, but not 7.0. For 7.5 these symlinks are not
      # necessary, and for 8.0 the cudatoolkit packages includes them.
      ln -T -s libcudart.so.7.0.28 $CONDA_PREFIX/lib/libcudart.so.7.0
      ln -T -s libcufft.so.7.0.35 $CONDA_PREFIX/lib/libcufft.so.7.0
    fi
    ;;
esac
