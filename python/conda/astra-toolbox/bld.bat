@echo off

set R=%SRC_DIR%

set B_BV=1_78
set B_BOOST=C:\local\boost_%B_BV%_0

cd /D "%B_BOOST%\lib64-msvc-14.1"

mkdir "%R%\lib\x64"
mkdir "%R%\bin\x64\Release_CUDA"

copy boost_unit_test_framework-vc141-mt-x64-%B_BV%.lib %R%\lib\x64
copy libboost_chrono-vc141-mt-x64-%B_BV%.lib %R%\lib\x64
copy libboost_date_time-vc141-mt-x64-%B_BV%.lib %R%\lib\x64
copy libboost_system-vc141-mt-x64-%B_BV%.lib %R%\lib\x64
copy libboost_thread-vc141-mt-x64-%B_BV%.lib %R%\lib\x64

cd %B_BOOST%

xcopy /i /e /q /y boost "%R%\lib\include\boost"

cd /D %R%

cd python

set CL=/DASTRA_CUDA /DASTRA_PYTHON "/I%R%\include" "/I%R%\lib\include" "/I%CUDA_PATH%\include"
copy "%LIBRARY_LIB%\AstraCuda64.lib" astra.lib
python builder.py build_ext --compiler=msvc install
