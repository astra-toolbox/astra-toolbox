@echo off
setlocal

set B_VC=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\
set B_CUDA_PATH=%CUDA_PATH_V12_9%

set ORIGIN=%CD%
set REPOROOT=%~dp0\..\..

call "%B_VC%\vcvars64.bat" 10.0.22621.0
if errorlevel 1 goto :error

cd /D "%REPOROOT%\build\msvc"
msbuild astra.sln /p:Configuration=Release_CUDA /p:Platform=x64 /t:astra /maxcpucount:20
if errorlevel 1 goto :error

cd /D "%REPOROOT%\python"
if not exist "%REPOROOT%\build\pypi\pkgs" mkdir "%REPOROOT%\build\pypi\pkgs"

set CL=/DASTRA_CUDA /DASTRA_BUILDING_CUDA /DASTRA_PYTHON /std:c++17
set INCLUDE=%REPOROOT%\include;%REPOROOT%\lib\include;%B_CUDA_PATH%\include;%INCLUDE%
set DISTUTILS_USE_SDK=1
set ASTRA_CONFIG=windows_cuda

copy ..\build\msvc\bin\x64\Release_CUDA\AstraCuda64.lib astra.lib
copy ..\build\msvc\bin\x64\Release_CUDA\AstraCuda64.dll astra

for %%P in (
    "C:\Python310"
    "C:\Python310"
    "C:\Python311"
    "C:\Python312"
    "C:\Python313"
    "C:\Python314"
) do (
    if exist ".\build" rmdir /s /q ".\build"
    "%%~P\python" -m pip uninstall -y astra-toolbox
    "%%~P\python" -m pip wheel --no-build-isolation --no-deps --no-cache-dir .
    if errorlevel 1 goto :error
    move "*.whl" "%REPOROOT%\build\pypi\pkgs\"
)

cd /d "%ORIGIN%"
endlocal
exit /b 0

:error
cd /d "%ORIGIN%"
endlocal
exit /b 1
