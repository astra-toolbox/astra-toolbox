@echo off

set R=%SRC_DIR%%

set B_VC=C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64
call "%B_VC%\vcvars64.bat"

cd /D %R%

set B_BV=1_61

mkdir "%R%\lib\x64"
mkdir "%R%\bin\x64\Release_CUDA"

cd /D %CONDA_PREFIX%\Library\lib


copy boost_unit_test_framework-vc140-mt-%B_BV%.lib %R%\lib\x64
copy libboost_chrono-vc140-mt-%B_BV%.lib %R%\lib\x64
copy libboost_date_time-vc140-mt-%B_BV%.lib %R%\lib\x64
copy libboost_system-vc140-mt-%B_BV%.lib %R%\lib\x64
copy libboost_thread-vc140-mt-%B_BV%.lib %R%\lib\x64

cd /D %CONDA_PREFIX%\Library\include

xcopy /i /e /q boost "%R%\lib\include\boost"

cd /D %R%

cd python

set VS90COMNTOOLS=%VS140COMNTOOLS%
set CL=/DASTRA_CUDA /DASTRA_PYTHON "/I%R%\include" "/I%R%\lib\include" "/I%CUDA_PATH%\include"
copy %CONDA_PREFIX%\Library\lib\AstraCuda64.lib astra.lib
python builder.py build_ext --compiler=msvc install
