@echo off

cd /D %~dp0
cd ..\..
set R=%CD%

call "%~dp0build_env.bat"

call "%B_VC%\vcvars64.bat" 10.0.22621.0

cd /D %~dp0

msbuild astra.sln /p:Configuration=Release_CUDA /p:Platform=x64 /t:astra /maxcpucount:20

cd /D %R%
cd python

IF EXIST build rd /s /q build
IF EXIST "%B_WINPYTHON3%\lib\site-packages\astra" rd /s /q "%B_WINPYTHON3%\lib\site-packages\astra"

set CL=/DASTRA_CUDA /DASTRA_BUILDING_CUDA /DASTRA_PYTHON /std:c++17
set INCLUDE=%R%\include;%R%\lib\include;%B_CUDA_PATH%\include;%INCLUDE%
set DISTUTILS_USE_SDK=1
set ASTRA_CONFIG=windows_cuda
copy ..\build\msvc\bin\x64\Release_CUDA\AstraCuda64.lib astra.lib
copy ..\build\msvc\bin\x64\Release_CUDA\AstraCuda64.dll astra
copy "%B_CUDA_PATH%\bin\cudart64_12.dll" astra
copy "%B_CUDA_PATH%\bin\cufft64_11.dll" astra
"%B_WINPYTHON3%\python" -m pip wheel --no-build-isolation --no-deps --no-cache-dir .

pause
