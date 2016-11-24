-----------------------------------------------------------------------
This file is part of the ASTRA Toolbox

Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
           2014-2015, CWI, Amsterdam
           http://visielab.uantwerpen.be/ and http://www.cwi.nl/
License: Open Source under GPLv3
Contact: astra@uantwerpen.be
Website: http://sf.net/projects/astra-toolbox
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
http://sf.net/projects/astra-toolbox .





Installation instructions:
---------------------------


Windows, binary:
-----------------

Add the mex and tools subdirectories to your matlab path and the Python module
to your Python path.



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





Windows, from source using Visual Studio 2008:
-----------------------------------------------

Requirements: Visual Studio 2008, boost, CUDA (driver+toolkit), matlab.
Note that a .zip with all required (and precompiled) boost files is
  available from our website.

Set the environment variable MATLAB_ROOT to your matlab install location.
Open astra_vc08.sln in Visual Studio.
Select the appropriate solution configuration.
  (typically Release_CUDA|win32 or Release_CUDA|x64)
Build the solution.
Install by copying AstraCuda32.dll or AstraCuda64.dll from bin/ and
  all .mexw32 or .mexw64 files from bin/Release_CUDA or bin/Debug_CUDA
  and the entire matlab/tools directory to a directory to be added to
  your matlab path.


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

