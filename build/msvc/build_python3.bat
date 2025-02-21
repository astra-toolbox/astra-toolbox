@echo off

cd /D %~dp0
cd ..\..
set R=%CD%

call "%~dp0build_env.bat"

call "%B_VC%\vcvars64.bat"

cd /D %~dp0

msbuild astra_vc14.sln /p:Configuration=Release_CUDA /p:Platform=x64 /t:astra_vc14 /maxcpucount:20

cd /D %R%
cd python

rd /s /q build
rd /s /q "%B_WINPYTHON3%\lib\site-packages\astra"

set CL=/DASTRA_CUDA /DASTRA_PYTHON /std:c++17
set INCLUDE=%R%\include;%R%\lib\include;%CUDA_PATH%\include;%INCLUDE%
copy ..\build\msvc\bin\x64\Release_CUDA\AstraCuda64.lib astra.lib
%B_WINPYTHON3%\python builder.py build_ext --compiler=msvc add_extra_lib --file=..\build\msvc\bin\x64\Release_CUDA\AstraCuda64.dll;"%CUDA_PATH_V12_8%\bin\cudart64_12.dll";"%CUDA_PATH_V12_8%\bin\cufft64_11.dll" bdist_wheel

pause
