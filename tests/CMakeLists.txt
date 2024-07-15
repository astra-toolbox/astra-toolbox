# Find libboost components ---------------------------------------------------

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.71 CONFIG REQUIRED COMPONENTS unit_test_framework)

# Collect source and header files --------------------------------------------

file(GLOB ASTRA_TEST_SOURCES "${CMAKE_SOURCE_DIR}/tests/*.cpp")

set(ASTRA_INCLUDES
    "${CMAKE_SOURCE_DIR}/include" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    "${CMAKE_SOURCE_DIR}/lib/include")

# Declare tests and register with cmake

add_executable(astratest ${ASTRA_TEST_SOURCES})

set_target_properties(
  astratest
  PROPERTIES POSITION_INDEPENDENT_CODE ON
             COMPILE_DEFINITIONS "ASTRA_CUDA"
             COMPILE_FEATURES cxx_std_11
             INCLUDE_DIRECTORIES "${ASTRA_INCLUDES}"
             LINK_LIBRARIES "Boost::unit_test_framework;astra")

add_test(COMMAND astratest)
