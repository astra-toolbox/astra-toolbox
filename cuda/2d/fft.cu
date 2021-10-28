/*
-----------------------------------------------------------------------
Copyright: 2010-2021, imec Vision Lab, University of Antwerp
           2014-2021, CWI, Amsterdam

Contact: astra@astra-toolbox.com
Website: http://www.astra-toolbox.com/

This file is part of the ASTRA Toolbox.


The ASTRA Toolbox is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The ASTRA Toolbox is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.

-----------------------------------------------------------------------
*/

#include "astra/cuda/2d/fft.h"
#include "astra/cuda/2d/util.h"

#include "astra/Logging.h"
#include "astra/Fourier.h"

#include <iostream>
#include <fstream>

#include <cufft.h>
#include <cuda.h>


using namespace astra;

// TODO: evaluate what we want to do in these situations:

#define CHECK_ERROR(errorMessage) do {                                     \
  cudaError_t err = cudaDeviceSynchronize();                               \
  if( cudaSuccess != err) {                                                \
      ASTRA_ERROR("Cuda error %s : %s",                                    \
              errorMessage,cudaGetErrorString( err));                      \
      exit(EXIT_FAILURE);                                                  \
  } } while (0)

#define SAFE_CALL( call) do {                                              \
  cudaError err = call;                                                    \
  if( cudaSuccess != err) {                                                \
      ASTRA_ERROR("Cuda error: %s ",                                       \
              cudaGetErrorString( err));                                   \
      exit(EXIT_FAILURE);                                                  \
  }                                                                        \
  err = cudaDeviceSynchronize();                                           \
  if( cudaSuccess != err) {                                                \
      ASTRA_ERROR("Cuda error: %s : ",                                     \
              cudaGetErrorString( err));                                   \
      exit(EXIT_FAILURE);                                                  \
  } } while (0)


namespace astraCUDA {

__global__ static void applyFilter_kernel(int _iProjectionCount,
                                          int _iFreqBinCount,
                                          cufftComplex * _pSinogram,
                                          cufftComplex * _pFilter)
{
	int iIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int iProjectionIndex = iIndex / _iFreqBinCount;

	if(iProjectionIndex >= _iProjectionCount)
	{
		return;
	}

	float fA = _pSinogram[iIndex].x;
	float fB = _pSinogram[iIndex].y;
	float fC = _pFilter[iIndex].x;
	float fD = _pFilter[iIndex].y;

	_pSinogram[iIndex].x = fA * fC - fB * fD;
	_pSinogram[iIndex].y = fA * fD + fC * fB;
}

__global__ static void rescaleInverseFourier_kernel(int _iProjectionCount,
                                                    int _iDetectorCount,
                                                    float* _pfInFourierOutput)
{
	int iIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int iProjectionIndex = iIndex / _iDetectorCount;
	int iDetectorIndex = iIndex % _iDetectorCount;

	if(iProjectionIndex >= _iProjectionCount)
	{
		return;
	}

	_pfInFourierOutput[iProjectionIndex * _iDetectorCount + iDetectorIndex] /= (float)_iDetectorCount;
}

static void rescaleInverseFourier(int _iProjectionCount, int _iDetectorCount,
                                  float * _pfInFourierOutput)
{
	const int iBlockSize = 256;
	int iElementCount = _iProjectionCount * _iDetectorCount;
	int iBlockCount = (iElementCount + iBlockSize - 1) / iBlockSize;

	rescaleInverseFourier_kernel<<< iBlockCount, iBlockSize >>>(_iProjectionCount,
	                                                            _iDetectorCount,
	                                                            _pfInFourierOutput);
	CHECK_ERROR("rescaleInverseFourier_kernel failed");
}

void applyFilter(int _iProjectionCount, int _iFreqBinCount,
                 cufftComplex * _pSinogram, cufftComplex * _pFilter)
{
	const int iBlockSize = 256;
	int iElementCount = _iProjectionCount * _iFreqBinCount;
	int iBlockCount = (iElementCount + iBlockSize - 1) / iBlockSize;

	applyFilter_kernel<<< iBlockCount, iBlockSize >>>(_iProjectionCount,
	                                                  _iFreqBinCount,
	                                                  _pSinogram, _pFilter);
	CHECK_ERROR("applyFilter_kernel failed");
}

static bool invokeCudaFFT(int _iProjectionCount, int _iDetectorCount,
                          const float * _pfDevSource,
                          cufftComplex * _pDevTargetComplex)
{
	cufftHandle plan;
	cufftResult result;

	result = cufftPlan1d(&plan, _iDetectorCount, CUFFT_R2C, _iProjectionCount);
	if(result != CUFFT_SUCCESS)
	{
		ASTRA_ERROR("Failed to plan 1d r2c fft");
		return false;
	}

	result = cufftExecR2C(plan, (cufftReal *)_pfDevSource, _pDevTargetComplex);
	cufftDestroy(plan);

	if(result != CUFFT_SUCCESS)
	{
		ASTRA_ERROR("Failed to exec 1d r2c fft");
		return false;
	}

	return true;
}

static bool invokeCudaIFFT(int _iProjectionCount, int _iDetectorCount,
                           const cufftComplex * _pDevSourceComplex,
                           float * _pfDevTarget)
{
	cufftHandle plan;
	cufftResult result;

	result = cufftPlan1d(&plan, _iDetectorCount, CUFFT_C2R, _iProjectionCount);
	if(result != CUFFT_SUCCESS)
	{
		ASTRA_ERROR("Failed to plan 1d c2r fft");
		return false;
	}

	// todo: why do we have to get rid of the const qualifier?
	result = cufftExecC2R(plan, (cufftComplex *)_pDevSourceComplex,
	                      (cufftReal *)_pfDevTarget);
	cufftDestroy(plan);

	if(result != CUFFT_SUCCESS)
	{
		ASTRA_ERROR("Failed to exec 1d c2r fft");
		return false;
	}

	return true;
}

bool allocateComplexOnDevice(int _iProjectionCount, int _iDetectorCount,
                             cufftComplex ** _ppDevComplex)
{
	size_t bufferSize = sizeof(cufftComplex) * _iProjectionCount * _iDetectorCount;
	SAFE_CALL(cudaMalloc((void **)_ppDevComplex, bufferSize));
	return true;
}

bool freeComplexOnDevice(cufftComplex * _pDevComplex)
{
	SAFE_CALL(cudaFree(_pDevComplex));
	return true;
}

bool uploadComplexArrayToDevice(int _iProjectionCount, int _iDetectorCount,
                                cufftComplex * _pHostComplexSource,
                                cufftComplex * _pDevComplexTarget)
{
	size_t memSize = sizeof(cufftComplex) * _iProjectionCount * _iDetectorCount;
	SAFE_CALL(cudaMemcpy(_pDevComplexTarget, _pHostComplexSource, memSize, cudaMemcpyHostToDevice));

	return true;
}

bool runCudaFFT(int _iProjectionCount, const float * _pfDevRealSource,
                int _iSourcePitch, int _iProjDets,
                int _iFFTRealDetectorCount, int _iFFTFourierDetectorCount,
                cufftComplex * _pDevTargetComplex)
{
	float * pfDevRealFFTSource = NULL;
	size_t bufferMemSize = sizeof(float) * _iProjectionCount * _iFFTRealDetectorCount;

	SAFE_CALL(cudaMalloc((void **)&pfDevRealFFTSource, bufferMemSize));
	SAFE_CALL(cudaMemset(pfDevRealFFTSource, 0, bufferMemSize));

	for(int iProjectionIndex = 0; iProjectionIndex < _iProjectionCount; iProjectionIndex++)
	{
		const float * pfSourceLocation = _pfDevRealSource + iProjectionIndex * _iSourcePitch;
		float * pfTargetLocation = pfDevRealFFTSource + iProjectionIndex * _iFFTRealDetectorCount;

		SAFE_CALL(cudaMemcpy(pfTargetLocation, pfSourceLocation, sizeof(float) * _iProjDets, cudaMemcpyDeviceToDevice));
	}

	bool bResult = invokeCudaFFT(_iProjectionCount, _iFFTRealDetectorCount,
	                             pfDevRealFFTSource, _pDevTargetComplex);
	if(!bResult)
	{
		return false;
	}

	SAFE_CALL(cudaFree(pfDevRealFFTSource));

	return true;
}

bool runCudaIFFT(int _iProjectionCount, const cufftComplex* _pDevSourceComplex,
                 float * _pfRealTarget,
                 int _iTargetPitch, int _iProjDets,
                 int _iFFTRealDetectorCount, int _iFFTFourierDetectorCount)
{
	float * pfDevRealFFTTarget = NULL;
	size_t bufferMemSize = sizeof(float) * _iProjectionCount * _iFFTRealDetectorCount;

	SAFE_CALL(cudaMalloc((void **)&pfDevRealFFTTarget, bufferMemSize));

	bool bResult = invokeCudaIFFT(_iProjectionCount, _iFFTRealDetectorCount,
	                              _pDevSourceComplex, pfDevRealFFTTarget);
	if(!bResult)
	{
		return false;
	}

	rescaleInverseFourier(_iProjectionCount, _iFFTRealDetectorCount,
	                      pfDevRealFFTTarget);

	SAFE_CALL(cudaMemset(_pfRealTarget, 0, sizeof(float) * _iProjectionCount * _iTargetPitch));

	for(int iProjectionIndex = 0; iProjectionIndex < _iProjectionCount; iProjectionIndex++)
	{
		const float * pfSourceLocation = pfDevRealFFTTarget + iProjectionIndex * _iFFTRealDetectorCount;
		float* pfTargetLocation = _pfRealTarget + iProjectionIndex * _iTargetPitch;

		SAFE_CALL(cudaMemcpy(pfTargetLocation, pfSourceLocation, sizeof(float) * _iProjDets, cudaMemcpyDeviceToDevice));
	}

	SAFE_CALL(cudaFree(pfDevRealFFTTarget));

	return true;
}

void genIdenFilter(int _iProjectionCount, cufftComplex * _pFilter,
                   int _iFFTRealDetectorCount, int _iFFTFourierDetectorCount)
{
	for(int iProjectionIndex = 0; iProjectionIndex < _iProjectionCount; iProjectionIndex++)
	{
		for(int iDetectorIndex = 0; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
		{
			int iIndex = iDetectorIndex + iProjectionIndex * _iFFTFourierDetectorCount;
			_pFilter[iIndex].x = 1.0f;
			_pFilter[iIndex].y = 0.0f;
		}
	}
}

void genCuFFTFilter(const SFilterConfig &_cfg, int _iProjectionCount,
               cufftComplex * _pFilter, int _iFFTRealDetectorCount,
               int _iFFTFourierDetectorCount)
{
	float * pfFilt = astra::genFilter(_cfg,
	                                  _iFFTRealDetectorCount,
	                                  _iFFTFourierDetectorCount);

	for(int iDetectorIndex = 0; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
	{
		float fFilterValue = pfFilt[iDetectorIndex];

		for(int iProjectionIndex = 0; iProjectionIndex < _iProjectionCount; iProjectionIndex++)
		{
			int iIndex = iDetectorIndex + iProjectionIndex * _iFFTFourierDetectorCount;
			_pFilter[iIndex].x = fFilterValue;
			_pFilter[iIndex].y = 0.0f;
		}
	}

	delete[] pfFilt;
}


}
