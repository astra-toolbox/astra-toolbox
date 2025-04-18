# The ASTRA Toolbox

The ASTRA Toolbox is a Python and MATLAB toolbox of high-performance GPU primitives for 2D and 3D tomography.

We support 2D parallel and fan beam geometries, and 3D parallel and cone beam.  All of them have highly flexible source/detector positioning.

A large number of 2D and 3D algorithms are available, including FBP, SIRT, SART, CGLS.

The basic forward and backward projection operations are GPU-accelerated, and directly callable from MATLAB and Python to enable building new algorithms.



## Documentation / samples

See the Python and MATLAB code samples in the `samples/` directory and https://www.astra-toolbox.com/docs/.


## Installation instructions

### Windows/Linux, using conda for Python packages

Requirements: [conda](https://conda.io/) Python environment, with 64 bit Python 3.9-3.13.

We provide packages for the ASTRA Toolbox in the `astra-toolbox` channel for the
conda package manager. We depend on CUDA packages available from the `nvidia`
channel. To install ASTRA into the current conda environement, run:

```
conda install -c astra-toolbox -c nvidia astra-toolbox
```

We also provide development packages between releases occasionally:

```
conda install -c astra-toolbox/label/dev -c nvidia astra-toolbox
```

### Linux, using pip for Python packages

Requirements: Python environment with 64 bit Python 3.9-3.13.

```
pip install astra-toolbox
```

Note that, unlike conda packages, we only provide packages built for Linux platform, and only with
a single reasonably recent version of CUDA toolkit. These packages depend on PyPI CUDA distribution
provided by NVIDIA.

### Windows, binary

Download and unpack .zip archive for the desired version from https://astra-toolbox.com/downloads/.
Add the mex\ and tools\ subdirectories to your MATLAB path, or install the Python
wheel using pip. We require the Microsoft Visual Studio 2017 redistributable
package. If this is not already installed on your system, it is included as
vc_redist.x64.exe in the ASTRA zip file.

### Linux, from source

Requirements: automake, libtool, g++ (7 or higher), CUDA (11.0 or higher)

Build dependencies can be obtained via the OS package manager or via conda. For example, a conda
environment with a full set of dependencies can be created with:

`conda create -n astra-build automake libtool gxx cuda-minimal-build libcufft-dev python cython scipy -c conda-forge`

You can then do `conda activate astra-build` and use `--with-cuda=$CONDA_PREFIX` in the build
configuration.

#### For MATLAB

Additional requirements: MATLAB (R2012a or higher)

```
cd build/linux
./autogen.sh   # when building a git version
./configure --with-cuda=/usr/local/cuda \
            --with-matlab=/usr/local/MATLAB/R2012a \
            --prefix=$HOME/astra \
            --with-install-type=module
make
make install
```
Add $HOME/astra/matlab and its subdirectories (tools, mex) to your MATLAB path.

If you want to build the Octave interface instead of the MATLAB interface,
specify `--enable-octave` instead of `--with-matlab=...`. The Octave files
will be installed into $HOME/astra/octave . On some Linux distributions
building the Astra Octave interface will require the Octave development package
to be installed (e.g., liboctave-dev on Ubuntu).


NB: Each MATLAB version only supports a specific range of g++ versions.
Despite this, if you have a newer g++ and if you get errors related to missing
GLIBCXX_3.4.xx symbols, it is often possible to work around this requirement
by deleting the version of libstdc++ supplied by MATLAB in
MATLAB_PATH/bin/glnx86 or MATLAB_PATH/bin/glnxa64 (at your own risk),
or setting `LD_PRELOAD=/usr/lib64/libstdc++.so.6` (or similar) when starting
MATLAB.

#### For Python

Additional requirements: Python (3.x), setuptools, Cython, scipy

```
cd build/linux
./autogen.sh   # when building a git version
./configure --with-cuda=/usr/local/cuda \
            --with-python \
            --with-install-type=module
make
make install
```

This will install Astra into your current Python environment.

#### As a C++ library

```
cd build/linux
./autogen.sh   # when building a git version
./configure --with-cuda=/usr/local/cuda
make
make install-dev
```

This will install the Astra library and C++ headers.


### macOS, from source

Use the Homebrew package manager to install libtool, autoconf, automake.

```
cd build/linux
./autogen.sh
CPPFLAGS="-I/usr/local/include" NVCCFLAGS="-I/usr/local/include" ./configure \
  --with-cuda=/usr/local/cuda \
  --with-matlab=/Applications/MATLAB_R2016b.app \
  --prefix=$HOME/astra \
  --with-install-type=module
make
make install
```

### Windows, from source using Visual Studio 2017

Requirements: Visual Studio 2017 (full or community),
              CUDA (11.0 or higher), MATLAB (R2012a or higher)
              and/or Python 3.x + setuptools + Cython + scipy.

#### Using the Visual Studio IDE:

1. Set the environment variable MATLAB_ROOT to your MATLAB install location.
2. Open build\msvc\astra_vc14.sln in Visual Studio.
3. Select the appropriate solution configuration (typically Release_CUDA|x64).
4. Build the solution.
5. Install by copying AstraCuda64.dll and all .mexw64 files from build\msvc\bin\x64\Release_CUDA
   and the entire matlab\tools directory to a directory to be added to your MATLAB path.


#### Using .bat scripts in build\msvc:

1. Edit build_env.bat and set up the correct library versions and paths.
2. * For MATLAB: Run build_matlab.bat. The .dll and .mexw64 files will be in build\msvc\bin\x64\Release_Cuda.
   * For Python: Run build_python3.bat. This will produce a Wheel file in python\dist directory, which can be installed using pip.


### Conda packages

See the instructions [here](build/conda/README.md).


## Testing your installation

To perform a (very) basic test of your ASTRA installation in Python, you can
run the following Python command.

```
import astra
astra.test()
```

To test your ASTRA installation in MATLAB, the equivalent command is:

```
astra_test
```

## References

If you use the ASTRA Toolbox for your research, we would appreciate it if you would refer to the following papers:

W. van Aarle, W. J. Palenstijn, J. Cant, E. Janssens, F. Bleichrodt, A. Dabravolski, J. De Beenhouwer, K. J. Batenburg, and J. Sijbers, “Fast and Flexible X-ray Tomography Using the ASTRA Toolbox”, Optics Express, 24(22), 25129-25147, (2016), https://dx.doi.org/10.1364/OE.24.025129

W. van Aarle, W. J. Palenstijn, J. De Beenhouwer, T. Altantzis, S. Bals, K. J. Batenburg, and J. Sijbers, “The ASTRA Toolbox: A platform for advanced algorithm development in electron tomography”, Ultramicroscopy, 157, 35–47, (2015), https://dx.doi.org/10.1016/j.ultramic.2015.05.002


Additionally, if you use parallel beam GPU code, we would appreciate it if you would refer to the following paper:

W. J. Palenstijn, K J. Batenburg, and J. Sijbers, "Performance improvements for iterative electron tomography reconstruction using graphics processing units (GPUs)", Journal of Structural Biology, vol. 176, issue 2, pp. 250-253, 2011, https://dx.doi.org/10.1016/j.jsb.2011.07.017


## License

The ASTRA Toolbox is open source under the GPLv3 license.

## Contact

email: astra@astra-toolbox.com
website: https://www.astra-toolbox.com/

Copyright: 2010-2024, imec Vision Lab, University of Antwerp
           2014-2024, CWI, Amsterdam
           https://visielab.uantwerpen.be/ and https://www.cwi.nl/
