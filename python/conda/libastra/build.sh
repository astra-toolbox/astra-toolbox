#!/bin/sh

cd $SRC_DIR/build/linux

$SRC_DIR/build/linux/autogen.sh

# Add C++11 to compiler flags if nvcc supports it, mostly to work around a boost bug
NVCC=$CUDA_ROOT/bin/nvcc
echo "int main(){return 0;}" > $CONDA_PREFIX/test.cu
$NVCC $CONDA_PREFIX/test.cu -ccbin $CC --std=c++11 -o $CONDA_PREFIX/test.out > /dev/null && EXTRA_NVCCFLAGS="--std=c++11" || true
rm -f $CONDA_PREFIX/test.out

$SRC_DIR/build/linux/configure --with-install-type=prefix --with-cuda=$CUDA_ROOT --prefix=$CONDA_PREFIX NVCCFLAGS="-ccbin $CC -I$CONDA_PREFIX/include $EXTRA_NVCCFLAGS" CC=$CC CXX=$CXX CPPFLAGS="-I$CONDA_PREFIX/include"

make install-libraries

LIBPATH=lib
if [ $ARCH == 64 ]
  then
    LIBPATH+=64
fi
cp -P $CUDA_ROOT/$LIBPATH/libcudart.so.* $CONDA_PREFIX/lib
cp -P $CUDA_ROOT/$LIBPATH/libcufft.so.* $CONDA_PREFIX/lib
