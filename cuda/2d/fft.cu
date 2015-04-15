/*
-----------------------------------------------------------------------
Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
           2014-2015, CWI, Amsterdam

Contact: astra@uantwerpen.be
Website: http://sf.net/projects/astra-toolbox

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
$Id$
*/

#include "fft.h"
#include "util.h"

#include <cufft.h>
#include <iostream>
#include <cuda.h>
#include <fstream>

#include "../../include/astra/Logging.h"

using namespace astra;

// TODO: evaluate what we want to do in these situations:

#define CHECK_ERROR(errorMessage) do {                                     \
  cudaError_t err = cudaThreadSynchronize();                               \
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
  err = cudaThreadSynchronize();                                           \
  if( cudaSuccess != err) {                                                \
      ASTRA_ERROR("Cuda error: %s : ",                                     \
              cudaGetErrorString( err));                                   \
      exit(EXIT_FAILURE);                                                  \
  } } while (0)


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


// Because the input is real, the Fourier transform is symmetric.
// CUFFT only outputs the first half (ignoring the redundant second half),
// and expects the same as input for the IFFT.
int calcFFTFourSize(int _iFFTRealSize)
{
	int iFFTFourSize = _iFFTRealSize / 2 + 1;

	return iFFTFourSize;
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

void genFilter(E_FBPFILTER _eFilter, float _fD, int _iProjectionCount,
               cufftComplex * _pFilter, int _iFFTRealDetectorCount,
               int _iFFTFourierDetectorCount, float _fParameter /* = -1.0f */)
{
	float * pfFilt = new float[_iFFTFourierDetectorCount];
	float * pfW = new float[_iFFTFourierDetectorCount];

	for(int iDetectorIndex = 0; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
	{
		float fRelIndex = (float)iDetectorIndex / (float)_iFFTRealDetectorCount;

		// filt = 2*( 0:(order/2) )./order;
		pfFilt[iDetectorIndex] = 2.0f * fRelIndex;
		//pfFilt[iDetectorIndex] = 1.0f;

		// w = 2*pi*(0:size(filt,2)-1)/order
		pfW[iDetectorIndex] = 3.1415f * 2.0f * fRelIndex;
	}

	switch(_eFilter)
	{
		case FILTER_RAMLAK:
		{
			// do nothing
			break;
		}
		case FILTER_SHEPPLOGAN:
		{
			// filt(2:end) = filt(2:end) .* (sin(w(2:end)/(2*d))./(w(2:end)/(2*d)))
			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				pfFilt[iDetectorIndex] = pfFilt[iDetectorIndex] * (sinf(pfW[iDetectorIndex] / 2.0f / _fD) / (pfW[iDetectorIndex] / 2.0f / _fD));
			}
			break;
		}
		case FILTER_COSINE:
		{
			// filt(2:end) = filt(2:end) .* cos(w(2:end)/(2*d))
			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				pfFilt[iDetectorIndex] = pfFilt[iDetectorIndex] * cosf(pfW[iDetectorIndex] / 2.0f / _fD);
			}
			break;
		}
		case FILTER_HAMMING:
		{
			// filt(2:end) = filt(2:end) .* (.54 + .46 * cos(w(2:end)/d))
			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				pfFilt[iDetectorIndex] = pfFilt[iDetectorIndex] * ( 0.54f + 0.46f * cosf(pfW[iDetectorIndex] / _fD));
			}
			break;
		}
		case FILTER_HANN:
		{
			// filt(2:end) = filt(2:end) .*(1+cos(w(2:end)./d)) / 2
			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				pfFilt[iDetectorIndex] = pfFilt[iDetectorIndex] * (1.0f + cosf(pfW[iDetectorIndex] / _fD)) / 2.0f;
			}
			break;
		}
		case FILTER_TUKEY:
		{
			float fAlpha = _fParameter;
			if(_fParameter < 0.0f) fAlpha = 0.5f;
			float fN = (float)_iFFTFourierDetectorCount;
			float fHalfN = fN / 2.0f;
			float fEnumTerm = fAlpha * fHalfN;
			float fDenom = (1.0f - fAlpha) * fHalfN;
			float fBlockStart = fHalfN - fEnumTerm;
			float fBlockEnd = fHalfN + fEnumTerm;

			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fAbsSmallN = fabs((float)iDetectorIndex);
				float fStoredValue = 0.0f;

				if((fBlockStart <= fAbsSmallN) && (fAbsSmallN <= fBlockEnd))
				{
					fStoredValue = 1.0f;
				}
				else
				{
					float fEnum = fAbsSmallN - fEnumTerm;
					float fCosInput = M_PI * fEnum / fDenom;
					fStoredValue = 0.5f * (1.0f + cosf(fCosInput));
				}

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		case FILTER_LANCZOS:
		{
			float fDenum = (float)(_iFFTFourierDetectorCount - 1);

			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fSmallN = (float)iDetectorIndex;
				float fX = 2.0f * fSmallN / fDenum - 1.0f;
				float fSinInput = M_PI * fX;
				float fStoredValue = 0.0f;

				if(fabsf(fSinInput) > 0.001f)
				{
					fStoredValue = sin(fSinInput)/fSinInput;
				}
				else
				{
					fStoredValue = 1.0f;
				}

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		case FILTER_TRIANGULAR:
		{
			float fNMinusOne = (float)(_iFFTFourierDetectorCount - 1);

			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fSmallN = (float)iDetectorIndex;
				float fAbsInput = fSmallN - fNMinusOne / 2.0f;
				float fParenInput = fNMinusOne / 2.0f - fabsf(fAbsInput);
				float fStoredValue = 2.0f / fNMinusOne * fParenInput;

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		case FILTER_GAUSSIAN:
		{
			float fSigma = _fParameter;
			if(_fParameter < 0.0f) fSigma = 0.4f;
			float fN = (float)_iFFTFourierDetectorCount;
			float fQuotient = (fN - 1.0f) / 2.0f;

			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fSmallN = (float)iDetectorIndex;
				float fEnum = fSmallN - fQuotient;
				float fDenom = fSigma * fQuotient;
				float fPower = -0.5f * (fEnum / fDenom) * (fEnum / fDenom);
				float fStoredValue = expf(fPower);

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		case FILTER_BARTLETTHANN:
		{
			const float fA0 = 0.62f;
			const float fA1 = 0.48f;
			const float fA2 = 0.38f;
			float fNMinusOne = (float)(_iFFTFourierDetectorCount) - 1.0f;

			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fSmallN = (float)iDetectorIndex;
				float fAbsInput = fSmallN / fNMinusOne - 0.5f;
				float fFirstTerm = fA1 * fabsf(fAbsInput);
				float fCosInput = 2.0f * M_PI * fSmallN / fNMinusOne;
				float fSecondTerm = fA2 * cosf(fCosInput);
				float fStoredValue = fA0 - fFirstTerm - fSecondTerm;

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		case FILTER_BLACKMAN:
		{
			float fAlpha = _fParameter;
			if(_fParameter < 0.0f) fAlpha = 0.16f;
			float fA0 = (1.0f - fAlpha) / 2.0f;
			float fA1 = 0.5f;
			float fA2 = fAlpha / 2.0f;
			float fNMinusOne = (float)(_iFFTFourierDetectorCount - 1);

			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fSmallN = (float)iDetectorIndex;
				float fCosInput1 = 2.0f * M_PI * 0.5f * fSmallN / fNMinusOne;
				float fCosInput2 = 4.0f * M_PI * 0.5f * fSmallN / fNMinusOne;
				float fStoredValue = fA0 - fA1 * cosf(fCosInput1) + fA2 * cosf(fCosInput2);

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		case FILTER_NUTTALL:
		{
			const float fA0 = 0.355768f;
			const float fA1 = 0.487396f;
			const float fA2 = 0.144232f;
			const float fA3 = 0.012604f;
			float fNMinusOne = (float)(_iFFTFourierDetectorCount) - 1.0f;

			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fSmallN = (float)iDetectorIndex;
				float fBaseCosInput = M_PI * fSmallN / fNMinusOne;
				float fFirstTerm = fA1 * cosf(2.0f * fBaseCosInput);
				float fSecondTerm = fA2 * cosf(4.0f * fBaseCosInput);
				float fThirdTerm = fA3 * cosf(6.0f * fBaseCosInput);
				float fStoredValue = fA0 - fFirstTerm + fSecondTerm - fThirdTerm;

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		case FILTER_BLACKMANHARRIS:
		{
			const float fA0 = 0.35875f;
			const float fA1 = 0.48829f;
			const float fA2 = 0.14128f;
			const float fA3 = 0.01168f;
			float fNMinusOne = (float)(_iFFTFourierDetectorCount) - 1.0f;

			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fSmallN = (float)iDetectorIndex;
				float fBaseCosInput = M_PI * fSmallN / fNMinusOne;
				float fFirstTerm = fA1 * cosf(2.0f * fBaseCosInput);
				float fSecondTerm = fA2 * cosf(4.0f * fBaseCosInput);
				float fThirdTerm = fA3 * cosf(6.0f * fBaseCosInput);
				float fStoredValue = fA0 - fFirstTerm + fSecondTerm - fThirdTerm;

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		case FILTER_BLACKMANNUTTALL:
		{
			const float fA0 = 0.3635819f;
			const float fA1 = 0.4891775f;
			const float fA2 = 0.1365995f;
			const float fA3 = 0.0106411f;
			float fNMinusOne = (float)(_iFFTFourierDetectorCount) - 1.0f;

			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fSmallN = (float)iDetectorIndex;
				float fBaseCosInput = M_PI * fSmallN / fNMinusOne;
				float fFirstTerm = fA1 * cosf(2.0f * fBaseCosInput);
				float fSecondTerm = fA2 * cosf(4.0f * fBaseCosInput);
				float fThirdTerm = fA3 * cosf(6.0f * fBaseCosInput);
				float fStoredValue = fA0 - fFirstTerm + fSecondTerm - fThirdTerm;

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		case FILTER_FLATTOP:
		{
			const float fA0 = 1.0f;
			const float fA1 = 1.93f;
			const float fA2 = 1.29f;
			const float fA3 = 0.388f;
			const float fA4 = 0.032f;
			float fNMinusOne = (float)(_iFFTFourierDetectorCount) - 1.0f;

			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fSmallN = (float)iDetectorIndex;
				float fBaseCosInput = M_PI * fSmallN / fNMinusOne;
				float fFirstTerm = fA1 * cosf(2.0f * fBaseCosInput);
				float fSecondTerm = fA2 * cosf(4.0f * fBaseCosInput);
				float fThirdTerm = fA3 * cosf(6.0f * fBaseCosInput);
				float fFourthTerm = fA4 * cosf(8.0f * fBaseCosInput);
				float fStoredValue = fA0 - fFirstTerm + fSecondTerm - fThirdTerm + fFourthTerm;

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		case FILTER_KAISER:
		{
			float fAlpha = _fParameter;
			if(_fParameter < 0.0f) fAlpha = 3.0f;
			float fPiTimesAlpha = M_PI * fAlpha;
			float fNMinusOne = (float)(_iFFTFourierDetectorCount - 1);
			float fDenom = (float)j0((double)fPiTimesAlpha);

			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fSmallN = (float)iDetectorIndex;
				float fSquareInput = 2.0f * fSmallN / fNMinusOne - 1;
				float fSqrtInput = 1.0f - fSquareInput * fSquareInput;
				float fBesselInput = fPiTimesAlpha * sqrt(fSqrtInput);
				float fEnum = (float)j0((double)fBesselInput);
				float fStoredValue = fEnum / fDenom;

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		case FILTER_PARZEN:
		{
			for(int iDetectorIndex = 1; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
			{
				float fSmallN = (float)iDetectorIndex;
				float fQ = fSmallN / (float)(_iFFTFourierDetectorCount - 1);
				float fStoredValue = 0.0f;

				if(fQ <= 0.5f)
				{
					fStoredValue = 1.0f - 6.0f * fQ * fQ * (1.0f - fQ);
				}
				else
				{
					float fCubedValue = 1.0f - fQ;
					fStoredValue = 2.0f * fCubedValue * fCubedValue * fCubedValue;
				}

				pfFilt[iDetectorIndex] *= fStoredValue;
			}

			break;
		}
		default:
		{
			ASTRA_ERROR("Cannot serve requested filter");
		}
	}

	// filt(w>pi*d) = 0;
	float fPiTimesD = M_PI * _fD;
	for(int iDetectorIndex = 0; iDetectorIndex < _iFFTFourierDetectorCount; iDetectorIndex++)
	{
		float fWValue = pfW[iDetectorIndex];

		if(fWValue > fPiTimesD)
		{
			pfFilt[iDetectorIndex] = 0.0f;
		}
	}

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
	delete[] pfW;
}

#ifdef STANDALONE

__global__ static void doubleFourierOutput_kernel(int _iProjectionCount,
                                                  int _iDetectorCount,
                                                  cufftComplex* _pFourierOutput)
{
	int iIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int iProjectionIndex = iIndex / _iDetectorCount;
	int iDetectorIndex = iIndex % _iDetectorCount;

	if(iProjectionIndex >= _iProjectionCount)
	{
		return;
	}

	if(iDetectorIndex <= (_iDetectorCount / 2))
	{
		return;
	}

	int iOtherDetectorIndex = _iDetectorCount - iDetectorIndex;

	_pFourierOutput[iProjectionIndex * _iDetectorCount + iDetectorIndex].x = _pFourierOutput[iProjectionIndex * _iDetectorCount + iOtherDetectorIndex].x;
	_pFourierOutput[iProjectionIndex * _iDetectorCount + iDetectorIndex].y = -_pFourierOutput[iProjectionIndex * _iDetectorCount + iOtherDetectorIndex].y;
}

static void doubleFourierOutput(int _iProjectionCount, int _iDetectorCount,
                                cufftComplex * _pFourierOutput)
{
	const int iBlockSize = 256;
	int iElementCount = _iProjectionCount * _iDetectorCount;
	int iBlockCount = (iElementCount + iBlockSize - 1) / iBlockSize;

	doubleFourierOutput_kernel<<< iBlockCount, iBlockSize >>>(_iProjectionCount,
	                                                          _iDetectorCount,
	                                                          _pFourierOutput);
	CHECK_ERROR("doubleFourierOutput_kernel failed");
}



static void writeToMatlabFile(const char * _fileName, const float * _pfData,
                              int _iRowCount, int _iColumnCount)
{
	std::ofstream out(_fileName);

	for(int iRowIndex = 0; iRowIndex < _iRowCount; iRowIndex++)
	{
		for(int iColumnIndex = 0; iColumnIndex < _iColumnCount; iColumnIndex++)
		{
			out << _pfData[iColumnIndex + iRowIndex * _iColumnCount] << " ";
		}

		out << std::endl;
	}
}

static void convertComplexToRealImg(const cufftComplex * _pComplex,
                                    int _iElementCount,
                                    float * _pfReal, float * _pfImaginary)
{
	for(int iIndex = 0; iIndex < _iElementCount; iIndex++)
	{
		_pfReal[iIndex] = _pComplex[iIndex].x;
		_pfImaginary[iIndex] = _pComplex[iIndex].y;
	}
}

void testCudaFFT()
{
	const int iProjectionCount = 2;
	const int iDetectorCount = 1024;
	const int iTotalElementCount = iProjectionCount * iDetectorCount;

	float * pfHostProj = new float[iTotalElementCount];
	memset(pfHostProj, 0, sizeof(float) * iTotalElementCount);

	for(int iProjectionIndex = 0; iProjectionIndex < iProjectionCount; iProjectionIndex++)
	{
		for(int iDetectorIndex = 0; iDetectorIndex < iDetectorCount; iDetectorIndex++)
		{
//			int

//			pfHostProj[iIndex] = (float)rand() / (float)RAND_MAX;
		}
	}

	writeToMatlabFile("proj.mat", pfHostProj, iProjectionCount, iDetectorCount);

	float * pfDevProj = NULL;
	SAFE_CALL(cudaMalloc((void **)&pfDevProj, sizeof(float) * iTotalElementCount));
	SAFE_CALL(cudaMemcpy(pfDevProj, pfHostProj, sizeof(float) * iTotalElementCount, cudaMemcpyHostToDevice));

	cufftComplex * pDevFourProj = NULL;
	SAFE_CALL(cudaMalloc((void **)&pDevFourProj, sizeof(cufftComplex) * iTotalElementCount));

	cufftHandle plan;
	cufftResult result;

	result = cufftPlan1d(&plan, iDetectorCount, CUFFT_R2C, iProjectionCount);
	if(result != CUFFT_SUCCESS)
	{
		ASTRA_ERROR("Failed to plan 1d r2c fft");
	}

	result = cufftExecR2C(plan, pfDevProj, pDevFourProj);
	if(result != CUFFT_SUCCESS)
	{
		ASTRA_ERROR("Failed to exec 1d r2c fft");
	}

	cufftDestroy(plan);

	doubleFourierOutput(iProjectionCount, iDetectorCount, pDevFourProj);

	cufftComplex * pHostFourProj = new cufftComplex[iTotalElementCount];
	SAFE_CALL(cudaMemcpy(pHostFourProj, pDevFourProj, sizeof(cufftComplex) * iTotalElementCount, cudaMemcpyDeviceToHost));

	float * pfHostFourProjReal = new float[iTotalElementCount];
	float * pfHostFourProjImaginary = new float[iTotalElementCount];

	convertComplexToRealImg(pHostFourProj, iTotalElementCount, pfHostFourProjReal, pfHostFourProjImaginary);

	writeToMatlabFile("proj_four_real.mat", pfHostFourProjReal, iProjectionCount, iDetectorCount);
	writeToMatlabFile("proj_four_imaginary.mat", pfHostFourProjImaginary, iProjectionCount, iDetectorCount);

	float * pfDevInFourProj = NULL;
	SAFE_CALL(cudaMalloc((void **)&pfDevInFourProj, sizeof(float) * iTotalElementCount));

	result = cufftPlan1d(&plan, iDetectorCount, CUFFT_C2R, iProjectionCount);
	if(result != CUFFT_SUCCESS)
	{
		ASTRA_ERROR("Failed to plan 1d c2r fft");
	}

	result = cufftExecC2R(plan, pDevFourProj, pfDevInFourProj);
	if(result != CUFFT_SUCCESS)
	{
		ASTRA_ERROR("Failed to exec 1d c2r fft");
	}

	cufftDestroy(plan);

	rescaleInverseFourier(iProjectionCount, iDetectorCount, pfDevInFourProj);

	float * pfHostInFourProj = new float[iTotalElementCount];
	SAFE_CALL(cudaMemcpy(pfHostInFourProj, pfDevInFourProj, sizeof(float) * iTotalElementCount, cudaMemcpyDeviceToHost));

	writeToMatlabFile("in_four.mat", pfHostInFourProj, iProjectionCount, iDetectorCount);

	SAFE_CALL(cudaFree(pDevFourProj));
	SAFE_CALL(cudaFree(pfDevProj));

	delete [] pfHostInFourProj;
	delete [] pfHostFourProjReal;
	delete [] pfHostFourProjImaginary;
	delete [] pfHostProj;
	delete [] pHostFourProj;
}

void downloadDebugFilterComplex(float * _pfHostSinogram, int _iProjectionCount,
                                int _iDetectorCount,
                                cufftComplex * _pDevFilter,
                                int _iFilterDetCount)
{
	cufftComplex * pHostFilter = NULL;
	size_t complMemSize = sizeof(cufftComplex) * _iFilterDetCount * _iProjectionCount;
	pHostFilter = (cufftComplex *)malloc(complMemSize);
	SAFE_CALL(cudaMemcpy(pHostFilter, _pDevFilter, complMemSize, cudaMemcpyDeviceToHost));

	for(int iTargetProjIndex = 0; iTargetProjIndex < _iProjectionCount; iTargetProjIndex++)
	{
		for(int iTargetDetIndex = 0; iTargetDetIndex < min(_iDetectorCount, _iFilterDetCount); iTargetDetIndex++)
		{
			cufftComplex source = pHostFilter[iTargetDetIndex + iTargetProjIndex * _iFilterDetCount];
			float fReadValue = sqrtf(source.x * source.x + source.y * source.y);
			_pfHostSinogram[iTargetDetIndex + iTargetProjIndex * _iDetectorCount] = fReadValue;
		}
	}

	free(pHostFilter);
}

void downloadDebugFilterReal(float * _pfHostSinogram, int _iProjectionCount,
                             int _iDetectorCount, float * _pfDevFilter,
                             int _iFilterDetCount)
{
	float * pfHostFilter = NULL;
	size_t memSize = sizeof(float) * _iFilterDetCount * _iProjectionCount;
	pfHostFilter = (float *)malloc(memSize);
	SAFE_CALL(cudaMemcpy(pfHostFilter, _pfDevFilter, memSize, cudaMemcpyDeviceToHost));

	for(int iTargetProjIndex = 0; iTargetProjIndex < _iProjectionCount; iTargetProjIndex++)
	{
		for(int iTargetDetIndex = 0; iTargetDetIndex < min(_iDetectorCount, _iFilterDetCount); iTargetDetIndex++)
		{
			float fSource = pfHostFilter[iTargetDetIndex + iTargetProjIndex * _iFilterDetCount];
			_pfHostSinogram[iTargetDetIndex + iTargetProjIndex * _iDetectorCount] = fSource;
		}
	}

	free(pfHostFilter);
}


#endif
