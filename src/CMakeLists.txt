# Find libboost components ---------------------------------------------------

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED)

# Collect source and header files --------------------------------------------
file(GLOB ASTRA_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp"
     "${CMAKE_SOURCE_DIR}/cuda/*/*.cu")

message(VERBOSE "Complete list of libastra sources: ${ASTRA_SOURCES}")

set(ASTRA_INCLUDES
    "${CMAKE_SOURCE_DIR}/include" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    "${CMAKE_SOURCE_DIR}/lib/include" ${Boost_INCLUDE_DIRS})

message(VERBOSE "Include directories for libastra: ${ASTRA_INCLUDES}")

# Declare libastra and assign all the properties -----------------------------

add_library(astra SHARED ${ASTRA_SOURCES})

target_compile_features(astra PRIVATE cxx_std_11)

set_target_properties(
  astra
  PROPERTIES POSITION_INDEPENDENT_CODE ON
             COMPILE_DEFINITIONS "ASTRA_CUDA"
             INCLUDE_DIRECTORIES "${ASTRA_INCLUDES}"
             LINK_LIBRARIES "${Boost_LIBRARIES}"
             VERSION "${CMAKE_PROJECT_VERSION}"
             SOVERSION "${CMAKE_PROJECT_VERSION_MAJOR}"
             CUDA_ARCHITECTURES 35
             37 50
             52 53
             60 61
             62 70
             72 75)
