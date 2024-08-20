@echo off

call "%~dp0build_env.bat"

cd /D %~dp0
cd ..\..
set R=%CD%

echo Removing bin directories
rd /s /q "%R%\build\msvc\bin\x64\Release_CUDA"

pause
