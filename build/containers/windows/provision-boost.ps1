#curl.exe -L -o boost.exe https://sourceforge.net/projects/boost/files/boost-binaries/1.78.0/boost_1_78_0-msvc-14.1-64.exe/download
#.\boost.exe /sp- /verysilent /suppressmsgboxes /norestart | more
#del boost.exe

choco install -y boost-msvc-14.1 --version=1.74.0
