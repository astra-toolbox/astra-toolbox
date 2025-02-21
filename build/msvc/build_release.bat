@echo off

cd /D %~dp0
cd ..\..

set R=%CD%

call "%~dp0build_env.bat"

cd /D %~dp0

rd /s /q release

mkdir release
cd release
mkdir matlab
mkdir python313

pause

cd %R%\build\msvc\release\matlab
mkdir astra-%B_RELEASE%
cd astra-%B_RELEASE%
xcopy /e /i %R%\samples\matlab samples
xcopy /e /i %R%\matlab\algorithms algorithms
xcopy /e /i %R%\matlab\tools tools
copy %R%\NEWS.txt .
copy %R%\README.txt .
copy %R%\COPYING COPYING.txt

copy %B_VCREDIST% .

mkdir mex
copy %R%\build\msvc\bin\x64\Release_CUDA\*.mexw64 mex
copy %R%\build\msvc\bin\x64\Release_CUDA\AstraCuda64.dll mex
copy %R%\build\msvc\bin\x64\Release_CUDA\AstraCuda64.lib mex
copy "%CUDA_PATH_V12_8%\bin\cudart64_12.dll" mex
copy "%CUDA_PATH_V12_8%\bin\cufft64_11.dll" mex

pause

rem -------------------------------------------------------------------

cd %R%\build\msvc\release\python313
mkdir astra-%B_RELEASE%
cd astra-%B_RELEASE%
xcopy /e /i %R%\samples\python samples
copy %R%\NEWS.txt .
copy %R%\COPYING COPYING.txt

copy %B_VCREDIST% .

copy %R%\python\dist\*.whl .

(
echo -----------------------------------------------------------------------
echo This file is part of the ASTRA Toolbox
echo.
echo Copyright: 2010-2024, imec Vision Lab, University of Antwerp
echo            2014-2024, CWI, Amsterdam
echo            https://visielab.uantwerpen.be/ and https://www.cwi.nl/
echo License: Open Source under GPLv3
echo Contact: astra@astra-toolbox.com
echo Website: https://www.astra-toolbox.com/
echo -----------------------------------------------------------------------
echo.
echo.
echo This directory contains pre-built Python modules for the ASTRA Toolbox.
echo.
echo It has been built with python %B_WP3%, installed via Chocolatey.
echo.
echo To use it, run 'pip3 install astra_toolbox-%B_RELEASE%-cp312-cp312-win_amd64.whl'
echo.
echo Sample code can be found in the samples\ directory.
) > README.txt

pause

cd %R%\build\msvc\release
%B_WINPYTHON3%\python -c "import shutil; shutil.make_archive('astra-%B_RELEASE%-matlab-win-x64', 'zip', 'matlab')"
%B_WINPYTHON3%\python -c "import shutil; shutil.make_archive('astra-%B_RELEASE%-python313-win-x64', 'zip', 'python313')"

pause
