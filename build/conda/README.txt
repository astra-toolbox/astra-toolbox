Building conda packages:

Linux:

Build container images by running the containers/setup*.sh scripts
./release.sh

Windows:

call C:\tools\miniconda3\condabin\activate.bat
Change to astra-toolbox\build\conda directory
# Build libastra packages, skipping the testing phase
conda build -m libastra\win64_build_config.yaml -c nvidia --no-test libastra
# Build and test astra-toolbox packages
conda build -m astra-toolbox\win64_build_config.yaml -c nvidia astra-toolbox
# Test the previously built libastra packages
conda build -c nvidia --test C:\tools\miniconda3\conda-bld\win-64\libastra*.conda
