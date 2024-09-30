@echo off

set R=%SRC_DIR%

cd /D %R%

cd python

set CL=/DASTRA_CUDA /DASTRA_PYTHON "/I%R%\include" "/I%R%\lib\include" "/I%CUDA_PATH%\include" /std:c++17
copy "%LIBRARY_LIB%\AstraCuda64.lib" astra.lib
python builder.py build_ext --compiler=msvc install
