Building conda packages
---

# Linux

Requirements: [podman](https://github.com/containers/podman) and [buildah](https://github.com/containers/buildah).

1. Change to `astra-toolbox/build/conda` directory
2. Build container images by running the `containers/setup*.sh` scripts
3. Run `./release.sh`

# Windows

Requirements: conda-build, git, [Visual Studio 2017](https://community.chocolatey.org/packages/visualstudio2017community) (for CUDA 11/12) and/or [Visual Studio 2022](https://community.chocolatey.org/packages/visualstudio2022community) (for CUDA 13), [Build Tools](https://community.chocolatey.org/packages/visualstudio2022buildtools) and [Native Desktop workload](https://community.chocolatey.org/packages/visualstudio2022-workload-nativedesktop), [Windows SDK version 10.0.22621.2](https://community.chocolatey.org/packages/windows-sdk-11-version-22H2-all) and/or Windows SDK version 10.0.26100.0, CUDA toolkit of desired version(s).

1. Activate conda: `C:\ProgramData\miniconda3\condabin\activate.bat`
2. You may also have to activate the VS Build Tools manually: `"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64 10.0.26100.0`
3. Change to `astra-toolbox\build\conda` directory
4. Build libastra packages, skipping the testing phase:
`conda build -m libastra\win64_build_config.yaml -c nvidia --no-test libastra`
5. Build and test astra-toolbox packages:
`conda build -m astra-toolbox\win64_build_config.yaml -c nvidia astra-toolbox`
6. Test the previously built libastra packages:
`conda build -c nvidia --test C:\ProgramData\miniconda3\conda-bld\win-64\libastra*.conda`

# Local installation

The built packages can be installed locally using `conda install astra-toolbox -c nvidia -c local`.
