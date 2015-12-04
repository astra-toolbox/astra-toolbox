Last update: 24-Nov-2015

REQUIREMENTS

**MPI**

  To be able to build and run the distributed ASTRA version you require a working 'MPI' 
  installation. This can either be installed using a package manager (e.g. install the
  openmpi or mpich package), or you can download and compile a version.  Below are 
  example steps to install the 'OpenMPI' implementation of the MPI API. The code will 
  be downloaded to a folder 'toolsSrc'  and the binaries will be placed in the folder 'tools'.


  $ mkdir toolsSrc 
  $ cd toolsSrc
  $ wget https://www.open-mpi.org/software/ompi/v1.8/downloads/openmpi-1.8.8.tar.gz
  $ tar xf openmpi-1.8.8.tar.gz
  $ cd openmpi-1.8.8
  $ ./configure --prefix=$HOME/tools
  $ make install
  $ export PATH=$HOME/tools/bin:$PATH
  $ export LD_LIBRARY_PATH=$HOME/tools/lib:$LD_LIBRARY_PATH

  Now you should be able to type 'mpirun' and get some output.

**Python**

  The distributed implementation is enabled through the use of Python. For this your 
  Python installation requires a number of extra packages. These can either be installed
  directly to the system Python folders using a package mananger or you can use a 
  virtual Python environment or use a Python distrubution such as miniconda. Here we 
  show both steps:

    *** Virtual Python Environment ***
      Creates a subfolder in the map ~/tools  where the Python files will be placed.

      $ cd ~/tools
      $ virtualenv astravenv2
      $ source astravenv2/bin/activate

      Install the following required Python modules:

      $ pip install numpy cython six mpi4py dill scipy

      Optional modules for figures:
      $ pip install matplotlib
      
      Optional for interactive terminal:
      $ pip install ipython

    *** miniconda ***

      $ cd ~/tools
      $ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
      $ sh ./Miniconda3-latest-Linux-x86_64.sh 
      $ << Follow the steps of the installer, in this example we install it in the folder: ~/miniconda3 >> 
      $ export PATH=~/tools/miniconda3/bin:$PATH  

      Install the required modules
      $ conda install numpy
      $ conda install cython
      $ conda install six
      $ conda install dill
      $ conda install scipy
      $ conda install matplotlib

      To make sure that the mpi4py package is compiled against your MPI version instead 
      of the one supplied by MiniConda use 'pip' to install the mpi4py module and 
      not the conda install * commands:
      $ pip install mpi4py
      
    

ASTRA Compilation & Installation

    After you've downloaded and extracted the distributed ASTRA version you have to use 
    the following options during configuration step.
    
      $ ./configure --with-cuda=/usr/local/cuda/ --prefix=$HOME/astra-toolbox/build_distributed/ CXX=mpicxx 
                    CC=mpicc --with-python --with-mpi=$HOME/tools/include/ 
    
      --with-mpi        #This enables MPI and will look for the include files in the default location
      --with-mpi=/path  #This enables MPI and will look for the 'mpi.h' include file in the given '/path' location
      CXX=mpicxx        #Change the C++ compiler to the MPI compiler (wrapper around g++)
      CC=mpicc          #Change the C   compiler to the MPI compiler (wrapper around gcc)
      --with-python     #Build the Python bindings for ASTRA
      --with-cuda=/path #Build the CUDA acceleration. This is required when using the distributed ASTRA Toolbox
      
      $ make; make install
      
      Add the installation paths to our environment variables:
      $ export LD_LIBRARY_PATH=$HOME/astra-toolbox/build_distributed/lib/:$LD_LIBRARY_PATH
      $ export PYTHONPATH=$HOME/astra-toolbox/build_distributed/python
      
      
      
Testing the ASTRA installation

    Now that the code is compiled and installed it is time to test if the installation 
    worked correctly. For this we will run a simple example script. This script will
    execute a SIRT reconstruction on multiple processes. 

      $ cd $HOME/astra-toolbox/mpi
      $ mpirun -np 2  ./toolbox.py --script ../samples/python/s007_3d_reconstruction_mpi.py


      -np 2       # The number of processes to launch, 2 in this example
      toolbox.py  # The ASTRA launch script, located in the  'mpi'  sub-folder. 
                    This script accepts the following arguments:
                    --script , this optional argument can point to the script that has 
                               to be executed. In this example it is the 
                               script 's007_3d_reconstruction_mpi.py' (also see 
                               'script changes' below).
                    -i , this optional argument indicates if the interpreter should be 
                          started after the script execution is finished.

      If no arguments are provided the program will present the user with an interactive 
      interpreter. This is either the built in Python interpreter or the IPython 
      interpreter if it is available.
    
      To specify on which machines to run you have to add extra arguments to mpirun. 
      See 'mpirun -h' for more info.
      
      
Script changes

    In order to create a proper domain distribution and let the work be distributed over 
    multiple nodes you have to  setup the mpi projector. This requires a projection and 
    a volume geometry as input which are then sub-divided over the available nodes. Use 
    the following syntax:
    
    import astra.mpi_c as mpi 
    proj_geom, vol_geom = mpi.create(proj_geom, vol_geom)

    Here 'proj_geom' is the projection geometry as returned by the 'create_proj_geom' 
    function and the 'vol_geom' is the volume geometry as returned by the 
    'create_vol_geom' function. If this is not done the MPI environment will not be 
    setup and the execution will not be distributed.
   
    All I/O, such as reading of files, creation of figures, etc. is done by the 
    root process (process 0).    

    
** Tips & Trick **

    Add the following two lines at the beginning of the script when running pyton scripts
    with plot commands when using a machine that is not running an X server. 
    
    import matplotlib
    matplotlib.use('agg')

      
      
      
      

    
