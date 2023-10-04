Building conda packages:

Linux:

Build container images by running the containers/setup*.sh scripts
./release.sh

Windows:

Open anaconda command prompt
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64 8.1 -vcvars_ver=14.0
Change to astra-toolbox\python\conda directory
conda build -m libastra\win64_build_config.yaml  libastra
conda build -m astra-toolbox\win64_build_config.yaml astra-toolbox
