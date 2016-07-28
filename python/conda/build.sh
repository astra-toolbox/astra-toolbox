cd build/linux
./autogen.sh
./configure --with-python --with-cuda=$CUDA_ROOT --prefix=$PREFIX
if [ $MAKEOPTS == '<UNDEFINED>' ]
  then
    MAKEOPTS=""
fi
make $MAKEOPTS python-root-install