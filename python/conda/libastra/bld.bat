@echo off

set R=%SRC_DIR%

set B_VC=C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64
call "%B_VC%\vcvars64.bat"

set B_BV=1_62
set B_BOOST=D:\wjp\boost_%B_BV%_0

cd /D "%B_BOOST%\lib64-msvc-14.0"

mkdir "%R%\lib\x64"
mkdir "%R%\bin\x64\Release_CUDA"

copy boost_unit_test_framework-vc140-mt-%B_BV%.lib %R%\lib\x64
copy libboost_chrono-vc140-mt-%B_BV%.lib %R%\lib\x64
copy libboost_date_time-vc140-mt-%B_BV%.lib %R%\lib\x64
copy libboost_system-vc140-mt-%B_BV%.lib %R%\lib\x64
copy libboost_thread-vc140-mt-%B_BV%.lib %R%\lib\x64

cd %B_BOOST%

xcopy /i /e /q boost "%R%\lib\include\boost"

cd /D %R%

msbuild astra_vc14.sln /p:Configuration=Release_CUDA /p:Platform=x64 /t:astra_vc14

copy bin\x64\Release_CUDA\AstraCuda64.dll "%CONDA_PREFIX%\Library\bin"
copy bin\x64\Release_CUDA\AstraCuda64.lib "%CONDA_PREFIX%\Library\lib"
copy "%CUDA_PATH%\bin\cudart64_80.dll" "%CONDA_PREFIX%\Library\bin"
copy "%CUDA_PATH%\bin\cufft64_80.dll" "%CONDA_PREFIX%\Library\bin"
