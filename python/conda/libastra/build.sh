cd build/linux
./autogen.sh
./configure --with-cuda=$CUDA_ROOT --prefix=$PREFIX
if [ $MAKEOPTS == '<UNDEFINED>' ]
  then
    MAKEOPTS=""
fi
make $MAKEOPTS install-libraries
LIBPATH=lib
if [ $ARCH == 64 ]
  then
    LIBPATH+=64
fi
cp -P $CUDA_ROOT/$LIBPATH/libcudart.so.* $PREFIX/lib
cp -P $CUDA_ROOT/$LIBPATH/libcufft.so.* $PREFIX/lib
