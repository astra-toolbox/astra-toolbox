@echo off

set R=%SRC_DIR%

cd /D %R%/build/msvc
python gen.py %cudatoolkit%

msbuild astra_vc14.sln /p:Configuration=Release_CUDA /p:Platform=x64 /t:astra_vc14 /maxcpucount:20

copy bin\x64\Release_CUDA\AstraCuda64.dll "%LIBRARY_BIN%"
copy bin\x64\Release_CUDA\AstraCuda64.lib "%LIBRARY_LIB%"
