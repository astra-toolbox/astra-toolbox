@echo off

call "%~dp0build_env.bat"

cd /D %~dp0
cd ..\..
set R=%CD%

echo Removing lib, bin directories
rd /s /q "%R%\lib\x64"
rd /s /q "%R%\lib\include\boost"
rd /s /q "%R%\bin\x64\Release_CUDA"

cd /D "%B_BOOST%\lib64-msvc-14.0"

mkdir "%R%\lib\x64"
mkdir "%R%\bin\x64\Release_CUDA"

echo Copying boost libraries
copy boost_unit_test_framework-vc140-mt-x64-%B_BV%.lib %R%\lib\x64
copy boost_unit_test_framework-vc140-mt-gd-x64-%B_BV%.lib %R%\lib\x64

copy libboost_chrono-vc140-mt-x64-%B_BV%.lib %R%\lib\x64
copy libboost_chrono-vc140-mt-gd-x64-%B_BV%.lib %R%\lib\x64

copy libboost_date_time-vc140-mt-x64-%B_BV%.lib %R%\lib\x64
copy libboost_date_time-vc140-mt-gd-x64-%B_BV%.lib %R%\lib\x64

copy libboost_system-vc140-mt-x64-%B_BV%.lib %R%\lib\x64
copy libboost_system-vc140-mt-gd-x64-%B_BV%.lib %R%\lib\x64

copy libboost_thread-vc140-mt-x64-%B_BV%.lib %R%\lib\x64
copy libboost_thread-vc140-mt-gd-x64-%B_BV%.lib %R%\lib\x64

cd %B_BOOST%

echo Copying boost headers

xcopy /i /e /q boost "%R%\lib\include\boost"

pause
