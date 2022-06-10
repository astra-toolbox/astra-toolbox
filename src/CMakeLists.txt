# Find libboost components ---------------------------------------------------

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.70 REQUIRED COMPONENTS thread)

# Collect source and header files --------------------------------------------
file(GLOB ASTRA_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

message(VERBOSE "Complete list of libastra sources: ${ASTRA_SOURCES}")

file(GLOB_RECURSE ASTRA_HEADERS "${CMAKE_SOURCE_DIR}/include/*"
     LIST_DIRECTORIES false)
message(VERBOSE "Public headers: ${ASTRA_HEADERS}")

set(ASTRA_INCLUDES "${CMAKE_SOURCE_DIR}/include"
                   "${CMAKE_SOURCE_DIR}/lib/include")

message(VERBOSE "Include directories for libastra: ${ASTRA_INCLUDES}")

# Declare libastra and assign all the properties -----------------------------

add_library(astra SHARED ${ASTRA_SOURCES})

if(WIN32)
  set(OUTPUT_NAME "AstraCuda64")
else()
  set(OUTPUT_NAME "astra")
endif()

set_target_properties(
  astra
  PROPERTIES POSITION_INDEPENDENT_CODE ON
             # DLL_EXPORTS marks functions as for external use (shared library)
             # on Windows
             # __SSE2__ enables Streaming SIMD Extensions 2 where available
             COMPILE_DEFINITIONS "DLL_EXPORTS;__SSE2__"
             COMPILE_FEATURES cxx_std_11
             INCLUDE_DIRECTORIES "${ASTRA_INCLUDES}"
             LINK_LIBRARIES Boost::thread
             VERSION 0.0.0
             SOVERSION 0
             OUTPUT_NAME "${OUTPUT_NAME}")

if(ASTRA_USE_CUDA)

  file(GLOB ASTRA_CUDA_SOURCES "${CMAKE_SOURCE_DIR}/cuda/*/*.cu")

  set(ASTRA_CUDA_INCLUDES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

  # Set CUDA target architectures --------------------------------------------

  if(NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES "")
    if((CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 5)
       AND (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11))
      list(APPEND CMAKE_CUDA_ARCHITECTURES "30;32")
    endif()
    if((CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 5)
       AND (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12))
      list(APPEND CMAKE_CUDA_ARCHITECTURES "35;37")
    endif()
    if((CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 6)
       AND (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12))
      list(APPEND CMAKE_CUDA_ARCHITECTURES "50;52;53")
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 8)
      list(APPEND CMAKE_CUDA_ARCHITECTURES "60;61;62")
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 9)
      list(APPEND CMAKE_CUDA_ARCHITECTURES "70;72")
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
      list(APPEND CMAKE_CUDA_ARCHITECTURES "75")
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.1)
      list(APPEND CMAKE_CUDA_ARCHITECTURES "80;86")
    endif()
  endif()

  message(
    STATUS
      "CUDA ${CMAKE_CUDA_COMPILER_VERSION} built for archs ${CMAKE_CUDA_ARCHITECTURES}"
  )

  set_property(TARGET astra PROPERTY CUDA_ARCHITECTURES
                                     ${CMAKE_CUDA_ARCHITECTURES})

  set_property(
    TARGET astra
    APPEND
    PROPERTY SOURCES ${ASTRA_CUDA_SOURCES})

  set_property(
    TARGET astra
    APPEND
    PROPERTY INCLUDE_DIRECTORIES ${ASTRA_CUDA_INCLUDES})

  # ASTRA_CUDA enables CUDA poritions of code
  set_property(
    TARGET astra
    APPEND
    PROPERTY COMPILE_DEFINITIONS "ASTRA_CUDA")

  set_property(
    TARGET astra
    APPEND
    PROPERTY LINK_LIBRARIES "cufft")

endif()
