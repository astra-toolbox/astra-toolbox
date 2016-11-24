@echo off

call "%~dp0build_env.bat"

call "%B_VC%\vcvars64.bat"

cd /D %~dp0
cd ..\..

set MATLAB_ROOT=%B_MATLAB_ROOT%

msbuild astra_vc14.sln /p:Configuration=Release_CUDA /p:Platform=x64

pause
