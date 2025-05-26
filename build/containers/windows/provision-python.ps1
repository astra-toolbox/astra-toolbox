choco install -y python39 --version=3.9.13
choco install -y python310 --version=3.10.11
choco install -y python311 --version=3.11.9
choco install -y python312 --version=3.12.4
choco install -y python313 --version=3.13.2

C:\python39\python -m pip install numpy scipy cython setuptools wheel
C:\python310\python -m pip install numpy scipy cython setuptools wheel
C:\python311\python -m pip install numpy scipy cython setuptools wheel
C:\python312\python -m pip install numpy scipy cython setuptools wheel
C:\python313\python -m pip install numpy scipy cython setuptools wheel

choco install -y miniconda3
C:\tools\miniconda3\shell\condabin\conda-hook.ps1
conda config --add channels defaults
conda activate base
conda install -y conda-build conda-verify
