# 3.11.0 is from scikit-build documentation
cmake_minimum_required(VERSION 3.11.0)

# Get consistent behavior out of all the compilers
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(
  astra-toolbox
  VERSION 1.9.9
  DESCRIPTION
    "a MATLAB and Python toolbox of high-performance GPU primitives for 2D and 3D tomography."
  LANGUAGES CXX CUDA)

add_subdirectory(src)

# Installs these targets to CMAKE_INSTALL_PREFIX
install(
  TARGETS astra
  EXPORT AstraTargets
  LIBRARY DESTINATION "lib"
)
install(
  DIRECTORY
  "${CMAKE_SOURCE_DIR}/include"
  DESTINATION ".")
install(
  EXPORT AstraTargets
  FILE "AstraTargets.cmake"
  NAMESPACE "astra::"
  DESTINATION "lib/cmake/astra")
