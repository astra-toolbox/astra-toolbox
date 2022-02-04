@echo off

set R=%SRC_DIR%

set B_BV=1_78
set B_BOOST=C:\local\boost_%B_BV%_0

cd /D "%B_BOOST%\lib64-msvc-14.1"

mkdir "%R%\lib\x64"
mkdir "%R%\bin\x64\Release_CUDA"

copy boost_unit_test_framework-vc141-mt-x64-%B_BV%.lib "%R%\lib\x64"
copy libboost_chrono-vc141-mt-x64-%B_BV%.lib "%R%\lib\x64"
copy libboost_date_time-vc141-mt-x64-%B_BV%.lib "%R%\lib\x64"
copy libboost_system-vc141-mt-x64-%B_BV%.lib "%R%\lib\x64"
copy libboost_thread-vc141-mt-x64-%B_BV%.lib "%R%\lib\x64"

cd %B_BOOST%

xcopy /i /e /q /y boost "%R%\lib\include\boost"

cd /D %R%/build/msvc
python gen.py vc14 %cudatoolkit%

cd /D %R%

msbuild astra_vc14.sln /p:Configuration=Release_CUDA /p:Platform=x64 /t:astra_vc14

copy bin\x64\Release_CUDA\AstraCuda64.dll "%LIBRARY_BIN%"
copy bin\x64\Release_CUDA\AstraCuda64.lib "%LIBRARY_LIB%"
