-----------------------------------------------------------------------
This file is part of the ASTRA Toolbox

Copyright: 2010-2018, imec Vision Lab, University of Antwerp
           2014-2018, CWI, Amsterdam
           http://visielab.uantwerpen.be/ and http://www.cwi.nl/
License: Open Source under GPLv3
Contact: astra@astra-toolbox.com
Website: http://www.astra-toolbox.com/
-----------------------------------------------------------------------


The ASTRA Toolbox is a MATLAB and Python toolbox of high-performance GPU
primitives for 2D and 3D tomography.

We support 2D parallel and fan beam geometries, and 3D parallel and cone beam.
All of them have highly flexible source/detector positioning.

A large number of 2D and 3D algorithms are available, including FBP, SIRT,
SART, CGLS.

The basic forward and backward projection operations are GPU-accelerated, and
directly callable from MATLAB and Python to enable building new algorithms.




Documentation / samples:
-------------------------

See the MATLAB and Python code samples in samples/ and on
http://www.astra-toolbox.com/ .





Installation instructions:
---------------------------


Windows, binary:
-----------------

Add the mex and tools subdirectories to your MATLAB path, or copy the Python
astra module to your Python site-packages directory. We require the Microsoft
Visual Studio 2015 redistributable package. If this is not already installed on
your system, it is included as vc_redist.x64.exe in the ASTRA zip file.



Linux/Windows, using conda for python
-------------------------------------

Requirements: `conda <http://conda.pydata.org/>`_ python environment, with 64 bit Python 2.7, 3.5 or 3.6.

There are packages available for the ASTRA Toolbox in the astra-toolbox
channel for the conda package manager. To use these, run the following
inside a conda environment.

conda install -c astra-toolbox astra-toolbox

conda install -c astra-toolbox/label/dev astra-toolbox



Linux, from source:
--------------------

For Matlab:

Requirements: g++, boost, CUDA (5.5 or higher), Matlab (R2012a or higher)

cd build/linux
./autogen.sh   # when building a git version
./configure --with-cuda=/usr/local/cuda \
            --with-matlab=/usr/local/MATLAB/R2012a \
            --prefix=$HOME/astra \
            --with-install-type=module
make
make install

Add $HOME/astra/matlab and its subdirectories (tools, mex) to your matlab path.

If you want to build the Octave interface instead of the Matlab interface,
specify --enable-octave instead of --with-matlab=... . The Octave files
will be installed into $HOME/astra/octave .


NB: Each matlab version only supports a specific range of g++ versions.
Despite this, if you have a newer g++ and if you get errors related to missing
GLIBCXX_3.4.xx symbols, it is often possible to work around this requirement
by deleting the version of libstdc++ supplied by matlab in
MATLAB_PATH/bin/glnx86 or MATLAB_PATH/bin/glnxa64 (at your own risk),
or setting LD_PRELOAD=/usr/lib64/libstdc++.so.6 (or similar) when starting
matlab.


For Python:

Requirements: g++, boost, CUDA (5.5 or higher), Python (2.7 or 3.x)

cd build/linux
./autogen.sh   # when building a git version
./configure --with-cuda=/usr/local/cuda \
            --with-python \
            --with-install-type=module
make
make install

This will install Astra into your current Python environment.


As a C++ library:

Requirements: g++, boost, CUDA (5.5 or higher)

cd build/linux
./autogen.sh   # when building a git version
./configure --with-cuda=/usr/local/cuda
make
make install-dev

This will install the Astra library and C++ headers.


macOS, from source:
--------------------

Use the Homebrew package manager to install boost, libtool, autoconf, automake.

cd build/linux
./autogen.sh
CPPFLAGS="-I/usr/local/include" NVCCFLAGS="-I/usr/local/include" ./configure \
  --with-cuda=/usr/local/cuda \
  --with-matlab=/Applications/MATLAB_R2016b.app \
  --prefix=$HOME/astra \
  --with-install-type=module
make
make install


Windows, from source using Visual Studio 2015:
-----------------------------------------------

Requirements: Visual Studio 2015 (full or community), boost (recent), CUDA 8.0,
              Matlab (R2012a or higher) and/or WinPython 2.7/3.x.

Using the Visual Studio IDE:

Set the environment variable MATLAB_ROOT to your matlab install location.
Copy boost headers to lib\include\boost, and boost libraries to lib\x64.
Open astra_vc14.sln in Visual Studio.
Select the appropriate solution configuration (typically Release_CUDA|x64).
Build the solution.
Install by copying AstraCuda64.dll and all .mexw64 files from
  bin\x64\Release_CUDA and the entire matlab/tools directory to a directory
  to be added to your matlab path.


Using .bat scripts in build\msvc:

Edit build_env.bat and set up the correct directories.
Run build_setup.bat to automatically copy the boost headers and libraries.
For matlab: Run build_matlab.bat. The .dll and .mexw64 files will be in bin\x64\Release_Cuda.
For python 2.7/3.5: Run build_python27.bat or build_python35.bat. Astra will be directly installed into site-packages.


Testing your installation:
---------------------------

To perform a (very) basic test of your ASTRA installation in Python, you can
run the following Python command.

import astra
astra.test()


To test your ASTRA installation in Matlab, the equivalent command is:

astra_test


References:
------------

If you use the ASTRA Toolbox for your research, we would appreciate it if you would refer to the following papers:

W. Van Aarle, W J. Palenstijn, J. Cant, E. Janssens, F. Bleichrodt, A. Dabravolski, J. De Beenhouwer, K. J. Batenburg, and J. Sijbers, "Fast and Flexible X-ray Tomography Using the ASTRA Toolbox", Optics Express, vol. 24, no. 22, pp. 25129-25147, 2016

W. Van Aarle, W J. Palenstijn, J. De Beenhouwer, T. Altantzis, S. Bals, K. J. Batenburg, and J. Sijbers, "The ASTRA Toolbox: a platform for advanced algorithm development in electron tomography", Ultramicroscopy, vol. 157, pp. 35–47, 2015

Additionally, if you use parallel beam GPU code, we would appreciate it if you would refer to the following paper:

W. J. Palenstijn, K J. Batenburg, and J. Sijbers, "Performance improvements
for iterative electron tomography reconstruction using graphics processing
units (GPUs)", Journal of Structural Biology, vol. 176, issue 2, pp. 250-253,
2011, http://dx.doi.org/10.1016/j.jsb.2011.07.017

