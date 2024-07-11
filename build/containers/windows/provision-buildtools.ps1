choco install -y visualstudio2017community
choco install -y visualstudio2017buildtools
choco install -y visualstudio2017-workload-nativedesktop
choco install -y visualstudio2017-workload-python
choco install -y git
choco install -y curl
choco install -y unzip
choco install -y windows-sdk-11-version-22H2-all --version=10.0.22621.2

curl.exe -L -o C:\Users\vagrant\vc_redist.x64.exe https://aka.ms/vs/17/release/vc_redist.x64.exe
