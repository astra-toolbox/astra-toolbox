# Find libboost components ---------------------------------------------------

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.70 REQUIRED COMPONENTS thread)

# Collect source and header files --------------------------------------------
file(GLOB ASTRA_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp"
     "${CMAKE_SOURCE_DIR}/cuda/*/*.cu")

message(VERBOSE "Complete list of libastra sources: ${ASTRA_SOURCES}")

file(GLOB_RECURSE ASTRA_HEADERS "${CMAKE_SOURCE_DIR}/include/*"
     LIST_DIRECTORIES false)
message(VERBOSE "Public headers: ${ASTRA_HEADERS}")

set(ASTRA_INCLUDES
    "${CMAKE_SOURCE_DIR}/include" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
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
             # ASTRA_CUDA enables CUDA poritions of code
             # DLL_EXPORTS marks functions as for external use (shared library) on Windows
             # __SSE2__ enables Streaming SIMD Extensions 2 where available
             COMPILE_DEFINITIONS "ASTRA_CUDA;DLL_EXPORTS;__SSE2__"
             COMPILE_FEATURES cxx_std_11
             INCLUDE_DIRECTORIES "${ASTRA_INCLUDES}"
             LINK_LIBRARIES "Boost::thread;cufft"
             VERSION 0.0.0
             SOVERSION 0
             OUTPUT_NAME "${OUTPUT_NAME}"
             CUDA_ARCHITECTURES 35
             37 50
             52 53
             60 61
             62 70
             72 75)
