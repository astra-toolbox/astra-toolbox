@echo off

cd /D %~dp0
cd ..\..
set R=%CD%

call "%~dp0build_env.bat"

call "%B_WINPYTHON3%\scripts\env.bat"
call "%B_VC%\vcvars64.bat"

cd /D %R%

msbuild astra_vc14.sln /p:Configuration=Release_CUDA /p:Platform=x64 /t:astra_vc14

cd python

rd /s /q build
rd /s /q "%WINPYDIR%\lib\site-packages\astra"

set CL=/DASTRA_CUDA /DASTRA_PYTHON
set INCLUDE=%R%\include;%R%\lib\include;%CUDA_PATH%\include
copy ..\bin\x64\Release_CUDA\AstraCuda64.lib astra.lib
python builder.py build_ext --compiler=msvc install
copy ..\bin\x64\Release_CUDA\AstraCuda64.dll "%WINPYDIR%\lib\site-packages\astra"

pause
