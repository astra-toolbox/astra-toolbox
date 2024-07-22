@echo off

call "%~dp0build_env.bat"

cd /D %~dp0
cd ..\..
set R=%CD%

echo Removing lib, bin directories
rd /s /q "%R%\lib\x64"
rd /s /q "%R%\lib\include\boost"
rd /s /q "%R%\build\msvc\bin\x64\Release_CUDA"

cd /D "%B_BOOST%\lib64-msvc-14.1"

mkdir "%R%\lib\x64"

echo Copying boost libraries
copy boost_unit_test_framework-vc141-mt-x64-%B_BV%.lib %R%\lib\x64
copy boost_unit_test_framework-vc141-mt-gd-x64-%B_BV%.lib %R%\lib\x64

copy libboost_chrono-vc141-mt-x64-%B_BV%.lib %R%\lib\x64
copy libboost_chrono-vc141-mt-gd-x64-%B_BV%.lib %R%\lib\x64

copy libboost_date_time-vc141-mt-x64-%B_BV%.lib %R%\lib\x64
copy libboost_date_time-vc141-mt-gd-x64-%B_BV%.lib %R%\lib\x64

copy libboost_system-vc141-mt-x64-%B_BV%.lib %R%\lib\x64
copy libboost_system-vc141-mt-gd-x64-%B_BV%.lib %R%\lib\x64

copy libboost_thread-vc141-mt-x64-%B_BV%.lib %R%\lib\x64
copy libboost_thread-vc141-mt-gd-x64-%B_BV%.lib %R%\lib\x64

cd %B_BOOST%

echo Copying boost headers

xcopy /i /e /q boost "%R%\lib\include\boost"

pause
