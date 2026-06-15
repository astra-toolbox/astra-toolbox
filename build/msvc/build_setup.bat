@echo off

call "%~dp0build_env.bat"

cd /D %~dp0

echo Removing bin directories
IF EXIST "bin\x64\Release_CUDA" rd /s /q "bin\x64\Release_CUDA"

pause
