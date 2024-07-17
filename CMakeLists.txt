# 3.11.0 is from scikit-build documentation 3.18.0 is required for
# CUDA_ARCHITECTURES property
cmake_minimum_required(VERSION 3.18.0)

# Get consistent behavior out of all the compilers
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(
  astra-toolbox
  VERSION 2.2.0
  DESCRIPTION
    "a MATLAB and Python toolbox of high-performance GPU primitives for 2D and 3D tomography."
  LANGUAGES CXX)

option(ASTRA_USE_CUDA "Whether libastra is built with CUDA." ON)

if(ASTRA_USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED COMPONENTS cudart cufft)
endif()

add_subdirectory(src)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(CTest)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
  add_subdirectory(tests)
endif()

# Installs these targets to CMAKE_INSTALL_PREFIX
install(
  TARGETS astra
  EXPORT AstraTargets
  LIBRARY DESTINATION "lib")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include" DESTINATION ".")
install(
  EXPORT AstraTargets
  FILE "AstraTargets.cmake"
  NAMESPACE "astra::"
  DESTINATION "lib/cmake/astra")

