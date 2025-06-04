#ifndef ASTRA_GPU_FFT_WRAPPER_H
#define ASTRA_GPU_FFT_WRAPPER_H

#ifdef ASTRA_BUILDING_CUDA
#include <cufft.h>
#endif

#ifdef ASTRA_BUILDING_HIP
#include <hipfft/hipfft.h>
#endif

#endif
