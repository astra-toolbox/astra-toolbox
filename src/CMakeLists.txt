# Find libboost components ---------------------------------------------------

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.70 CONFIG REQUIRED COMPONENTS thread )

# Collect source and header files --------------------------------------------
set(ASTRA_SOURCES
    Algorithm.cpp
    ArtAlgorithm.cpp
    AstraObjectFactory.cpp
    AstraObjectManager.cpp
    BackProjectionAlgorithm.cpp
    CglsAlgorithm.cpp
    CompositeGeometryManager.cpp
    ConeProjectionGeometry3D.cpp
    ConeVecProjectionGeometry3D.cpp
    Config.cpp
    DataProjector.cpp
    DataProjectorPolicies.cpp
    FanFlatBeamLineKernelProjector2D.cpp
    FanFlatBeamStripKernelProjector2D.cpp
    FanFlatProjectionGeometry2D.cpp
    FanFlatVecProjectionGeometry2D.cpp
    Features.cpp
    FilteredBackProjectionAlgorithm.cpp
    Filters.cpp
    Float32Data.cpp
    Float32Data2D.cpp
    Float32Data3D.cpp
    Float32Data3DMemory.cpp
    Float32ProjectionData2D.cpp
    Float32ProjectionData3D.cpp
    Float32ProjectionData3DMemory.cpp
    Float32VolumeData2D.cpp
    Float32VolumeData3D.cpp
    Float32VolumeData3DMemory.cpp
    ForwardProjectionAlgorithm.cpp
    Fourier.cpp
    GeometryUtil2D.cpp
    GeometryUtil3D.cpp
    Globals.cpp
    Logging.cpp
    ParallelBeamBlobKernelProjector2D.cpp
    ParallelBeamDistanceDrivenProjector2D.cpp
    ParallelBeamLinearKernelProjector2D.cpp
    ParallelBeamLineKernelProjector2D.cpp
    ParallelBeamStripKernelProjector2D.cpp
    ParallelProjectionGeometry2D.cpp
    ParallelProjectionGeometry3D.cpp
    ParallelVecProjectionGeometry2D.cpp
    ParallelVecProjectionGeometry3D.cpp
    PlatformDepSystemCode.cpp
    PluginAlgorithmFactory.cpp
    ProjectionGeometry2D.cpp
    ProjectionGeometry3D.cpp
    Projector2D.cpp
    Projector3D.cpp
    ReconstructionAlgorithm2D.cpp
    ReconstructionAlgorithm3D.cpp
    SartAlgorithm.cpp
    SheppLogan.cpp
    SirtAlgorithm.cpp
    SparseMatrix.cpp
    SparseMatrixProjectionGeometry2D.cpp
    SparseMatrixProjector2D.cpp
    Utilities.cpp
    VolumeGeometry2D.cpp
    VolumeGeometry3D.cpp
    XMLDocument.cpp
    XMLNode.cpp)

file(GLOB_RECURSE ASTRA_HEADERS "${CMAKE_SOURCE_DIR}/include/*"
     LIST_DIRECTORIES false)
message(VERBOSE "Public headers: ${ASTRA_HEADERS}")

set(ASTRA_INCLUDES "${CMAKE_SOURCE_DIR}/include"
                   "${CMAKE_SOURCE_DIR}/lib/include")

message(VERBOSE "Include directories for libastra: ${ASTRA_INCLUDES}")

# Declare libastra and assign all the properties -----------------------------

add_library(astra SHARED ${ASTRA_SOURCES})

set_target_properties(
  astra
  PROPERTIES POSITION_INDEPENDENT_CODE ON
             COMPILE_FEATURES cxx_std_11
             INCLUDE_DIRECTORIES "${ASTRA_INCLUDES}"
             LINK_LIBRARIES Boost::thread
             VERSION 0.0.0
             SOVERSION 0
             OUTPUT_NAME "${OUTPUT_NAME}")

if(MSVC OR WIN32)
  # DLL_EXPORTS marks functions as for external use (shared library) on Windows
  target_compile_definitions(
    astra
    PRIVATE
    "DLL_EXPORTS"
  )
endif(MSVC OR WIN32)

if(ASTRA_USE_CUDA)

  set(ASTRA_CUDA_SOURCES
      CudaProjector2D.cpp
      CudaProjector3D.cpp
      CudaReconstructionAlgorithm2D.cpp
      CudaBackProjectionAlgorithm.cpp
      CudaDartMaskAlgorithm.cpp
      CudaDartMaskAlgorithm3D.cpp
      CudaDataOperationAlgorithm.cpp
      CudaRoiSelectAlgorithm.cpp
      CudaDartSmoothingAlgorithm.cpp
      CudaDartSmoothingAlgorithm3D.cpp
      CudaFilteredBackProjectionAlgorithm.cpp
      CudaForwardProjectionAlgorithm.cpp
      CudaSartAlgorithm.cpp
      CudaSirtAlgorithm.cpp
      CudaCglsAlgorithm.cpp
      CudaCglsAlgorithm3D.cpp
      CudaEMAlgorithm.cpp
      CudaFDKAlgorithm3D.cpp
      CudaSirtAlgorithm3D.cpp
      CudaBackProjectionAlgorithm3D.cpp
      CudaForwardProjectionAlgorithm3D.cpp
      Float32Data3DGPU.cpp
      Float32ProjectionData3DGPU.cpp
      Float32VolumeData3DGPU.cpp
      "${CMAKE_SOURCE_DIR}/cuda/2d/algo.cu"
      "${CMAKE_SOURCE_DIR}/cuda/2d/par_fp.cu"
      "${CMAKE_SOURCE_DIR}/cuda/2d/par_bp.cu"
      "${CMAKE_SOURCE_DIR}/cuda/2d/fan_fp.cu"
      "${CMAKE_SOURCE_DIR}/cuda/2d/fan_bp.cu"
      "${CMAKE_SOURCE_DIR}/cuda/2d/fbp.cu"
      "${CMAKE_SOURCE_DIR}/cuda/2d/sirt.cu"
      "${CMAKE_SOURCE_DIR}/cuda/2d/sart.cu"
      "${CMAKE_SOURCE_DIR}/cuda/2d/cgls.cu"
      "${CMAKE_SOURCE_DIR}/cuda/2d/em.cu"
      "${CMAKE_SOURCE_DIR}/cuda/2d/astra.cu"
      "${CMAKE_SOURCE_DIR}/cuda/2d/util.cu"
      "${CMAKE_SOURCE_DIR}/cuda/2d/arith.cu"
      "${CMAKE_SOURCE_DIR}/cuda/2d/fft.cu"
      "${CMAKE_SOURCE_DIR}/cuda/2d/darthelper.cu"
      "${CMAKE_SOURCE_DIR}/cuda/3d/darthelper3d.cu"
      "${CMAKE_SOURCE_DIR}/cuda/3d/algo3d.cu"
      "${CMAKE_SOURCE_DIR}/cuda/3d/cgls3d.cu"
      "${CMAKE_SOURCE_DIR}/cuda/3d/cone_fp.cu"
      "${CMAKE_SOURCE_DIR}/cuda/3d/cone_bp.cu"
      "${CMAKE_SOURCE_DIR}/cuda/3d/fdk.cu"
      "${CMAKE_SOURCE_DIR}/cuda/3d/par3d_fp.cu"
      "${CMAKE_SOURCE_DIR}/cuda/3d/par3d_bp.cu"
      "${CMAKE_SOURCE_DIR}/cuda/3d/sirt3d.cu"
      "${CMAKE_SOURCE_DIR}/cuda/3d/astra3d.cu"
      "${CMAKE_SOURCE_DIR}/cuda/3d/util3d.cu"
      "${CMAKE_SOURCE_DIR}/cuda/3d/arith3d.cu"
      "${CMAKE_SOURCE_DIR}/cuda/3d/mem3d.cu")

  set(ASTRA_CUDA_INCLUDES ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

  target_sources(
    astra
    PRIVATE
    ${ASTRA_CUDA_SOURCES})

  target_include_directories(
    astra
    PUBLIC
    ${ASTRA_CUDA_INCLUDES})

  # ASTRA_CUDA enables CUDA portions of code
  target_compile_definitions(
    astra
    PUBLIC
    "ASTRA_CUDA")

  target_link_libraries(
    astra
    PUBLIC
    CUDA::cufft CUDA::cudart)

endif()
