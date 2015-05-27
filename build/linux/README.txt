Requirements: g++, boost, CUDA (driver+toolkit), Matlab and/or Python (2.7 or 3.x)

cd build/linux
./autogen.sh   # when building a git version
./configure --with-cuda=/usr/local/cuda \
            --with-matlab=/usr/local/MATLAB/R2012a \
            --with-python \
            --prefix=/usr/local/astra
make
make install
Add /usr/local/astra/lib to your LD_LIBRARY_PATH.
Add /usr/local/astra/matlab and its subdirectories (tools, mex)
  to your matlab path.
Add /usr/local/astra/python to your PYTHONPATH.


NB: Each matlab version only supports a specific range of g++ versions.
Despite this, if you have a newer g++ and if you get errors related to missing
GLIBCXX_3.4.xx symbols, it is often possible to work around this requirement
by deleting the version of libstdc++ supplied by matlab in
MATLAB_PATH/bin/glnx86 or MATLAB_PATH/bin/glnxa64 (at your own risk).

