@echo off

cd /D %~dp0
cd ..\..

set R=%CD%

call "%~dp0build_env.bat"

rd /s /q release

mkdir release
cd release
mkdir src
mkdir matlab
mkdir python27
mkdir python36

cd src
git clone -b %B_RELEASE_TAG% https://github.com/astra-toolbox/astra-toolbox astra-%B_RELEASE%
cd astra-%B_RELEASE%
rd /s /q .git

pause

cd %R%\release\matlab
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
copy %R%\bin\x64\Release_CUDA\*.mexw64 mex
copy %R%\bin\x64\Release_CUDA\AstraCuda64.dll mex
copy %R%\bin\x64\Release_CUDA\AstraCuda64.lib mex
copy "%CUDA_PATH%\bin\cudart64_80.dll" mex
copy "%CUDA_PATH%\bin\cufft64_80.dll" mex

pause

rem -------------------------------------------------------------------

cd %R%\release\python27
mkdir astra-%B_RELEASE%
cd astra-%B_RELEASE%
xcopy /e /i %R%\samples\python samples
copy %R%\NEWS.txt .
copy %R%\COPYING COPYING.txt

copy %B_VCREDIST% .

mkdir astra
call "%B_WINPYTHON2%\scripts\env.bat"
copy %WINPYDIR%\lib\site-packages\astra\*.* astra
copy %R%\bin\x64\Release_CUDA\AstraCuda64.lib astra
copy "%CUDA_PATH%\bin\cudart64_80.dll" astra
copy "%CUDA_PATH%\bin\cufft64_80.dll" astra

(
echo -----------------------------------------------------------------------
echo This file is part of the ASTRA Toolbox
echo.
echo Copyright: 2010-2018, imec Vision Lab, University of Antwerp
echo            2014-2018, CWI, Amsterdam
echo            http://visielab.uantwerpen.be/ and http://www.cwi.nl/
echo License: Open Source under GPLv3
echo Contact: astra@astra-toolbox.com
echo Website: http://www.astra-toolbox.com/
echo -----------------------------------------------------------------------
echo.
echo.
echo This directory contains pre-built Python modules for the ASTRA Toolbox.
echo.
echo It has been built with WinPython-64bit-%B_WP2%.
echo.
echo To use it, move the astra\ directory to your existing site-packages directory.
echo Its exact location depends on your Python installation, but should look
echo similar to %B_README_WP2% .
echo.
echo Sample code can be found in the samples\ directory.
) > README.txt

pause

rem -------------------------------------------------------------------

cd %R%\release\python36
mkdir astra-%B_RELEASE%
cd astra-%B_RELEASE%
xcopy /e /i %R%\samples\python samples
copy %R%\NEWS.txt .
copy %R%\COPYING COPYING.txt

copy %B_VCREDIST% .

mkdir astra
call "%B_WINPYTHON3%\scripts\env.bat"
copy %WINPYDIR%\lib\site-packages\astra\*.* astra
copy %R%\bin\x64\Release_CUDA\AstraCuda64.lib astra
copy "%CUDA_PATH%\bin\cudart64_80.dll" astra
copy "%CUDA_PATH%\bin\cufft64_80.dll" astra

(
echo -----------------------------------------------------------------------
echo This file is part of the ASTRA Toolbox
echo.
echo Copyright: 2010-2018, imec Vision Lab, University of Antwerp
echo            2014-2018, CWI, Amsterdam
echo            http://visielab.uantwerpen.be/ and http://www.cwi.nl/
echo License: Open Source under GPLv3
echo Contact: astra@astra-toolbox.com
echo Website: http://www.astra-toolbox.com/
echo -----------------------------------------------------------------------
echo.
echo.
echo This directory contains pre-built Python modules for the ASTRA Toolbox.
echo.
echo It has been built with WinPython-64bit-%B_WP3%.
echo.
echo To use it, move the astra\ directory to your existing site-packages directory.
echo Its exact location depends on your Python installation, but should look
echo similar to %B_README_WP3% .
echo.
echo Sample code can be found in the samples\ directory.
) > README.txt

pause

cd %R%\release
python -c "import shutil; shutil.make_archive('astra-%B_RELEASE%-matlab-win-x64', 'zip', 'matlab')"
python -c "import shutil; shutil.make_archive('astra-%B_RELEASE%-python27-win-x64', 'zip', 'python27')"
python -c "import shutil; shutil.make_archive('astra-%B_RELEASE%-python36-win-x64', 'zip', 'python36')"
python -c "import shutil; shutil.make_archive('astra-%B_RELEASE%', 'zip', 'src')"

pause
