@echo off

call "%~dp0build_env.bat"

call "%B_VC%\vcvars64.bat"

cd %~dp0
cd ..\..

msbuild astra_vc14.sln /p:Configuration=Release_CUDA /p:Platform=x64 /t:clean

pause
