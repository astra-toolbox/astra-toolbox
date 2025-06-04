#ifndef ASTRA_GPU_RUNTIME_WRAPPER_H
#define ASTRA_GPU_RUNTIME_WRAPPER_H

#ifdef ASTRA_BUILDING_CUDA
#include <cuda.h>
#endif

#ifdef ASTRA_BUILDING_HIP
#include <hip/hip_runtime.h>

// Namespaces
// TODO: Enable this when supporting CUDA and HIP in the same binary
//#define astraCUDA astraHIP
//#define astraCUDA3d astraHIP3d


// This list is semi-auto-generated based on hipify-perl -examine output.

// CUDA functions / constants
#define cudaAddressModeBorder hipAddressModeBorder
#define cudaAddressModeClamp hipAddressModeClamp
#define cudaArray hipArray
#define cudaChannelFormatDesc hipChannelFormatDesc
#define cudaChannelFormatKindFloat hipChannelFormatKindFloat
#define cudaCreateChannelDesc hipCreateChannelDesc
#define cudaCreateTextureObject hipCreateTextureObject
#define cudaDestroyTextureObject hipDestroyTextureObject
#define cudaDeviceProp hipDeviceProp_t
#define cudaErrorSetOnActiveProcess hipErrorSetOnActiveProcess
#define cudaError_t hipError_t
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventRecord hipEventRecord
#define cudaEvent_t hipEvent_t
#define cudaExtent hipExtent
#define cudaFilterModeLinear hipFilterModeLinear
#define cudaFreeArray hipFreeArray
#define cudaFree hipFree
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc3DArray hipMalloc3DArray
#define cudaMalloc3D hipMalloc3D
#define cudaMallocArray hipMallocArray
#define cudaMalloc hipMalloc
#define cudaMallocPitch hipMallocPitch
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpy2D hipMemcpy2D
#define cudaMemcpy2DToArrayAsync hipMemcpy2DToArrayAsync
#define cudaMemcpy3DAsync hipMemcpy3DAsync
#define cudaMemcpy3D hipMemcpy3D
#define cudaMemcpy3DParms hipMemcpy3DParms
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemset2DAsync hipMemset2DAsync
#define cudaMemset2D hipMemset2D
#define cudaMemset3D hipMemset3D
#define cudaMemsetAsync hipMemsetAsync
#define cudaPitchedPtr hipPitchedPtr
#define cudaPos hipPos
#define cudaReadModeElementType hipReadModeElementType
#define cudaResourceDesc hipResourceDesc
#define cudaResourceTypeArray hipResourceTypeArray
#define cudaResourceTypePitch2D hipResourceTypePitch2D
#define cudaSetDevice hipSetDevice
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStream_t hipStream_t
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaSuccess hipSuccess
#define cudaTextureAddressMode hipTextureAddressMode
#define cudaTextureDesc hipTextureDesc
#define cudaTextureObject_t hipTextureObject_t
#define CUFFT_C2R HIPFFT_C2R
#define cufftComplex hipfftComplex
#define cufftDestroy hipfftDestroy
#define cufftExecC2R hipfftExecC2R
#define cufftExecR2C hipfftExecR2C
#define cufftHandle hipfftHandle
#define cufftPlan1d hipfftPlan1d
#define CUFFT_R2C HIPFFT_R2C
#define cufftReal hipfftReal
#define cufftResult hipfftResult
#define cufftSetStream hipfftSetStream
#define CUFFT_SUCCESS HIPFFT_SUCCESS
#define make_cudaExtent make_hipExtent
#define make_cudaPos make_hipPos

// Special case
#define cudaMemcpyToSymbolAsync(S,...) hipMemcpyToSymbolAsync(HIP_SYMBOL(S),__VA_ARGS__)

#endif

#endif

