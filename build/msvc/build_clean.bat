@echo off

set MATLAB_ROOT=C:\Program Files\MATLAB\R2015a

call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64\vcvars64.bat"

cd %~dp0
cd ..\..

msbuild astra_vc14.sln /p:Configuration=Release_CUDA /p:Platform=x64 /t:clean

pause
