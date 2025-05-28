/*
-----------------------------------------------------------------------
Copyright: 2010-2022, imec Vision Lab, University of Antwerp
           2014-2022, CWI, Amsterdam

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

#include "astra/cuda/gpu_runtime_wrapper.h"
#include "astra/cuda/gpu_fft_wrapper.h"

#include "astra/cuda/2d/fft.h"
#include "astra/cuda/2d/util.h"

#include "astra/Logging.h"
#include "astra/Fourier.h"


using namespace astra;

namespace astraCUDA {

bool checkCufft(cufftResult err, const char *msg)
{
	if (err != CUFFT_SUCCESS) {
		ASTRA_ERROR("%s: CUFFT error %d.", msg, err);
		return false;
	} else {
		return true;
	}
}

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

__global__ static void applyFilter_singleFilter_kernel(int _iProjectionCount,
                                          int _iFreqBinCount,
                                          cufftComplex * _pSinogram,
                                          cufftComplex * _pFilter)
{
	int iIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int iProjectionIndex = iIndex / _iFreqBinCount;
	int iFilterIndex = iIndex % _iFreqBinCount;

	if(iProjectionIndex >= _iProjectionCount)
	{
		return;
	}

	float fA = _pSinogram[iIndex].x;
	float fB = _pSinogram[iIndex].y;
	float fC = _pFilter[iFilterIndex].x;
	float fD = _pFilter[iFilterIndex].y;

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

bool rescaleInverseFourier(int _iProjectionCount, int _iDetectorCount,
                           float * _pfInFourierOutput,
                           std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

	const int iBlockSize = 256;
	int iElementCount = _iProjectionCount * _iDetectorCount;
	int iBlockCount = (iElementCount + iBlockSize - 1) / iBlockSize;

	rescaleInverseFourier_kernel<<< iBlockCount, iBlockSize, 0, stream() >>>(_iProjectionCount,
	                                                            _iDetectorCount,
	                                                            _pfInFourierOutput);

	return stream.syncIfSync("rescaleInverseFourier");
}

bool applyFilter(int _iProjectionCount, int _iFreqBinCount,
                 cufftComplex * _pSinogram, cufftComplex * _pFilter,
                 bool singleFilter,
                 std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

	const int iBlockSize = 256;
	int iElementCount = _iProjectionCount * _iFreqBinCount;
	int iBlockCount = (iElementCount + iBlockSize - 1) / iBlockSize;

	if (singleFilter) {
		applyFilter_singleFilter_kernel<<< iBlockCount, iBlockSize, 0, stream() >>>(_iProjectionCount,
		                                                  _iFreqBinCount,
		                                                  _pSinogram, _pFilter);
	} else {
		applyFilter_kernel<<< iBlockCount, iBlockSize, 0, stream() >>>(_iProjectionCount,
		                                                  _iFreqBinCount,
		                                                  _pSinogram, _pFilter);

	}

	return stream.syncIfSync("applyFilter");
}

static bool invokeCudaFFT(int _iProjectionCount, int _iDetectorCount,
                          const float * _pfDevSource,
                          cufftComplex * _pDevTargetComplex,
                          cudaStream_t stream)
{
	cufftHandle plan;

	if (!checkCufft(cufftPlan1d(&plan, _iDetectorCount, CUFFT_R2C, _iProjectionCount), "invokeCudaFFT plan")) {
		return false;
	}

	if (!checkCufft(cufftSetStream(plan, stream), "invokeCudaFFT plan stream")) {
		cufftDestroy(plan);
		return false;
	}

	if (!checkCufft(cufftExecR2C(plan, (cufftReal *)_pfDevSource, _pDevTargetComplex), "invokeCudaFFT exec")) {
		cufftDestroy(plan);
		return false;
	}

	if (!checkCuda(cudaStreamSynchronize(stream), "invokeCudaFFT sync")) {
		cufftDestroy(plan);
		return false;
	}

	cufftDestroy(plan);
	return true;
}

static bool invokeCudaIFFT(int _iProjectionCount, int _iDetectorCount,
                           const cufftComplex * _pDevSourceComplex,
                           float * _pfDevTarget,
                           cudaStream_t stream)
{
	cufftHandle plan;

	if (!checkCufft(cufftPlan1d(&plan, _iDetectorCount, CUFFT_C2R, _iProjectionCount), "invokeCudaIFFT plan")) {
		return false;
	}

	if (!checkCufft(cufftSetStream(plan, stream), "invokeCudaIFFT plan stream")) {
		cufftDestroy(plan);
		return false;
	}

	// Getting rid of the const qualifier is due to cufft API issue?
	if (!checkCufft(cufftExecC2R(plan, (cufftComplex *)_pDevSourceComplex,
	                      (cufftReal *)_pfDevTarget), "invokeCudaIFFT exec"))
	{
		cufftDestroy(plan);
		return false;
	}

	if (!checkCuda(cudaStreamSynchronize(stream), "invokeCudaIFFT sync")) {
		cufftDestroy(plan);
		return false;
	}

	cufftDestroy(plan);
	return true;
}

bool allocateComplexOnDevice(int _iProjectionCount, int _iDetectorCount,
                             cufftComplex ** _ppDevComplex)
{
	size_t bufferSize = sizeof(cufftComplex) * _iProjectionCount * _iDetectorCount;
	return checkCuda(cudaMalloc((void **)_ppDevComplex, bufferSize), "fft allocateComplexOnDevice");
}

bool freeComplexOnDevice(cufftComplex * _pDevComplex)
{
	return checkCuda(cudaFree(_pDevComplex), "fft freeComplexOnDevice");
}

bool uploadComplexArrayToDevice(int _iProjectionCount, int _iDetectorCount,
                                cufftComplex * _pHostComplexSource,
                                cufftComplex * _pDevComplexTarget,
                                std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

	size_t memSize = sizeof(cufftComplex) * _iProjectionCount * _iDetectorCount;
	bool ok = checkCuda(cudaMemcpyAsync(_pDevComplexTarget, _pHostComplexSource, memSize, cudaMemcpyHostToDevice, stream()), "fft uploadComplexArrayToDevice");

	ok &= stream.syncIfSync("fft uploadComplexArrayToDevice");
	return ok;
}

bool runCudaFFT(int _iProjectionCount,
                const float * D_pfSource, int _iSourcePitch,
                int _iProjDets, int _iPaddedSize,
                cufftComplex * D_pcTarget,
                std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

	float * D_pfPaddedSource = NULL;
	size_t bufferMemSize = sizeof(float) * _iProjectionCount * _iPaddedSize;

	if (!checkCuda(cudaMalloc((void **)&D_pfPaddedSource, bufferMemSize), "runCudaFFT malloc")) {
		return false;
	}
	if (!checkCuda(cudaMemsetAsync(D_pfPaddedSource, 0, bufferMemSize, stream()), "runCudaFFT memset")) {
		cudaFree(D_pfPaddedSource);
		return false;
	}

	// pitched memcpy 2D to handle both source pitch and target padding
	if (!checkCuda(cudaMemcpy2DAsync(D_pfPaddedSource, _iPaddedSize*sizeof(float), D_pfSource, _iSourcePitch*sizeof(float), _iProjDets*sizeof(float), _iProjectionCount, cudaMemcpyDeviceToDevice, stream()), "runCudaFFT memcpy")) {
		cudaFree(D_pfPaddedSource);
		return false;
	}

	if (!invokeCudaFFT(_iProjectionCount, _iPaddedSize, D_pfPaddedSource, D_pcTarget, stream())) {
		cudaFree(D_pfPaddedSource);
		return false;
	}

	if (!stream.sync("runCudaFFT sync")) {
		cudaFree(D_pfPaddedSource);
		return false;
	}

	cudaFree(D_pfPaddedSource);
	return true;
}

bool runCudaIFFT(int _iProjectionCount, const cufftComplex *D_pcSource,
                 float * D_pfTarget, int _iTargetPitch,
                 int _iProjDets, int _iPaddedSize,
                 std::optional<cudaStream_t> _stream)
{
	StreamHelper stream(_stream);
	if (!stream)
		return false;

	float * D_pfPaddedTarget = NULL;
	size_t bufferMemSize = sizeof(float) * _iProjectionCount * _iPaddedSize;

	if (!checkCuda(cudaMalloc((void **)&D_pfPaddedTarget, bufferMemSize), "runCudaIFFT malloc")) {
		return false;
	}

	if (!invokeCudaIFFT(_iProjectionCount, _iPaddedSize,
	                    D_pcSource, D_pfPaddedTarget, stream()))
	{
		cudaFree(D_pfPaddedTarget);
		return false;
	}

	rescaleInverseFourier(_iProjectionCount, _iPaddedSize,
	                      D_pfPaddedTarget, stream());

	if (!checkCuda(cudaMemsetAsync(D_pfTarget, 0, sizeof(float) * _iProjectionCount * _iTargetPitch, stream()), "runCudaIFFT memset")) {
		cudaFree(D_pfPaddedTarget);
		return false;
	}

	// pitched memcpy 2D to handle both source padding and target pitch
	if (!checkCuda(cudaMemcpy2DAsync(D_pfTarget, _iTargetPitch*sizeof(float), D_pfPaddedTarget, _iPaddedSize*sizeof(float), _iProjDets*sizeof(float), _iProjectionCount, cudaMemcpyDeviceToDevice, stream()), "runCudaIFFT memcpy")) {
		cudaFree(D_pfPaddedTarget);
		return false;
	}

	if (!stream.sync("runCudaIFFT sync")) {
		cudaFree(D_pfPaddedTarget);
		return false;
	}

	cudaFree(D_pfPaddedTarget);
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

bool prepareCuFFTFilter(const SFilterConfig &cfg,
                        cufftComplex *&D_filter,
                        bool &singleFilter,
                        int iProjectionCount, int iDetectorCount,
                        std::optional<cudaStream_t> _stream)
{
	D_filter = nullptr;
	singleFilter = false;

	StreamHelper stream(_stream);
	if (!stream)
		return false;

	if (cfg.m_eType == astra::FILTER_NONE)
		return true;

	if (cfg.m_eType != astra::FILTER_SINOGRAM && cfg.m_eType != astra::FILTER_RSINOGRAM)
		singleFilter = true;

	int filterRows;
	if (singleFilter)
		filterRows = 1;
	else
		filterRows = iProjectionCount;

	int iPaddedDetCount = calcNextPowerOfTwo(2 * iDetectorCount);
	int iHalfFFTSize = astra::calcFFTFourierSize(iPaddedDetCount);
	//int iFFTRealDetCount = astra::calcNextPowerOfTwo(2 * dims.iProjDets);
	//int iFreqBinCount = astra::calcFFTFourierSize(iFFTRealDetCount);

	size_t filterSize = (size_t)filterRows * iHalfFFTSize;

	if (!allocateComplexOnDevice(filterRows, iHalfFFTSize, &D_filter)) {
		D_filter = nullptr;
		return false;
	}

	std::vector<cufftComplex> hostFilter(filterSize);

	switch(cfg.m_eType)
	{
		case astra::FILTER_NONE:
			// handled above
			break;
		case astra::FILTER_RAMLAK:
		case astra::FILTER_SHEPPLOGAN:
		case astra::FILTER_COSINE:
		case astra::FILTER_HAMMING:
		case astra::FILTER_HANN:
		case astra::FILTER_TUKEY:
		case astra::FILTER_LANCZOS:
		case astra::FILTER_TRIANGULAR:
		case astra::FILTER_GAUSSIAN:
		case astra::FILTER_BARTLETTHANN:
		case astra::FILTER_BLACKMAN:
		case astra::FILTER_NUTTALL:
		case astra::FILTER_BLACKMANHARRIS:
		case astra::FILTER_BLACKMANNUTTALL:
		case astra::FILTER_FLATTOP:
		case astra::FILTER_KAISER:
		case astra::FILTER_PARZEN:
		{
			genCuFFTFilter(cfg, filterRows, &hostFilter[0], iPaddedDetCount, iHalfFFTSize);
			bool ok = uploadComplexArrayToDevice(filterRows, iHalfFFTSize, &hostFilter[0], D_filter, stream());
			ok &= stream.syncIfSync("prepareCuFFTFilter upload");
			if (!ok) {
				cudaFree(D_filter);
				D_filter = nullptr;
				return false;
			}
			break;
		}
		case astra::FILTER_PROJECTION:
		{
			// make sure the offered filter has the correct size
			assert(cfg.m_iCustomFilterWidth == iHalfFFTSize);
			assert(cfg.m_iCustomFilterHeight == 1);

			for (int i = 0; i < iHalfFFTSize; ++i)
			{
				float fValue = cfg.m_pfCustomFilter[i];

				for (int j = 0; j < filterRows; ++j)
				{
					hostFilter[i + j * iHalfFFTSize].x = fValue;
					hostFilter[i + j * iHalfFFTSize].y = 0.0f;
				}
			}
			bool ok = uploadComplexArrayToDevice(filterRows, iHalfFFTSize, &hostFilter[0], D_filter, stream());
			ok &= stream.syncIfSync("prepareCuFFTFilter upload");
			if (!ok) {
				cudaFree(D_filter);
				D_filter = nullptr;
				return false;
			}
			break;
		}
		case astra::FILTER_SINOGRAM:
		{
			// make sure the offered filter has the correct size
			assert(cfg.m_iCustomFilterWidth == iHalfFFTSize);
			assert(cfg.m_iCustomFilterHeight == iProjectionCount);
			assert(filterRows == iProjectionCount);

			for (int i = 0; i < iHalfFFTSize; ++i)
			{
				for (int j = 0; j < filterRows; ++j)
				{
					float fValue = cfg.m_pfCustomFilter[i + j * iHalfFFTSize];

					hostFilter[i + j * iHalfFFTSize].x = fValue;
					hostFilter[i + j * iHalfFFTSize].y = 0.0f;
				}
			}
			bool ok = uploadComplexArrayToDevice(filterRows, iHalfFFTSize, &hostFilter[0], D_filter, stream());
			ok &= stream.syncIfSync("prepareCuFFTFilter upload");
			if (!ok) {
				cudaFree(D_filter);
				D_filter = nullptr;
				return false;
			}
			break;
		}
		case astra::FILTER_RPROJECTION:
		{
			size_t iSpatialFilterSize = filterRows * iPaddedDetCount;
			std::vector<float> hostSpatialFilter(iSpatialFilterSize);

			int iUsedFilterWidth = min(cfg.m_iCustomFilterWidth, iPaddedDetCount);
			int iStartFilterIndex = (cfg.m_iCustomFilterWidth - iUsedFilterWidth) / 2;
			int iMaxFilterIndex = iStartFilterIndex + iUsedFilterWidth;

			int iFilterShiftSize = cfg.m_iCustomFilterWidth / 2;

			for (int iDetectorIndex = iStartFilterIndex; iDetectorIndex < iMaxFilterIndex; iDetectorIndex++)
			{
				int iFFTInFilterIndex = (iDetectorIndex + iPaddedDetCount - iFilterShiftSize) % iPaddedDetCount;
				float fValue = cfg.m_pfCustomFilter[iDetectorIndex];

				for (int iProjectionIndex = 0; iProjectionIndex < filterRows; iProjectionIndex++)
				{
					hostSpatialFilter[iFFTInFilterIndex + iProjectionIndex * iPaddedDetCount] = fValue;
				}
			}

			float* D_spatialFilter = NULL;
			if (!checkCuda(cudaMalloc((void **)&D_spatialFilter, sizeof(float) * iSpatialFilterSize), "prepareCuFFTFilter malloc")) {
				cudaFree(D_filter);
				D_filter = nullptr;
				return false;
			}
			if (!checkCuda(cudaMemcpy(D_spatialFilter, &hostSpatialFilter[0], sizeof(float) * iSpatialFilterSize, cudaMemcpyHostToDevice), "prepareCuFFTFilter memcpy")) {
				cudaFree(D_filter);
				D_filter = nullptr;
				return false;
			}

			bool ok = runCudaFFT(filterRows, D_spatialFilter, iPaddedDetCount, iPaddedDetCount, iPaddedDetCount, D_filter, stream());

			// need to synchronize here for the cudaFree
			ok &= stream.sync("prepareCuFFTFilter FFT");

			cudaFree(D_spatialFilter);

			if (!ok) {
				cudaFree(D_filter);
				D_filter = nullptr;
				return false;
			}
			break;
		}
		case astra::FILTER_RSINOGRAM:
		{
			size_t iSpatialFilterSize = filterRows * iPaddedDetCount;
			std::vector<float> hostSpatialFilter(iSpatialFilterSize);

			int iUsedFilterWidth = min(cfg.m_iCustomFilterWidth, iPaddedDetCount);
			int iStartFilterIndex = (cfg.m_iCustomFilterWidth - iUsedFilterWidth) / 2;
			int iMaxFilterIndex = iStartFilterIndex + iUsedFilterWidth;

			int iFilterShiftSize = cfg.m_iCustomFilterWidth / 2;

			for(int iDetectorIndex = iStartFilterIndex; iDetectorIndex < iMaxFilterIndex; iDetectorIndex++)
			{
				int iFFTInFilterIndex = (iDetectorIndex + iPaddedDetCount - iFilterShiftSize) % iPaddedDetCount;

				for(int iProjectionIndex = 0; iProjectionIndex < iProjectionCount; iProjectionIndex++)
				{
					float fValue = cfg.m_pfCustomFilter[iDetectorIndex + iProjectionIndex * cfg.m_iCustomFilterWidth];
					hostSpatialFilter[iFFTInFilterIndex + iProjectionIndex * iPaddedDetCount] = fValue;
				}
			}

			float* D_spatialFilter = NULL;
			if (!checkCuda(cudaMalloc((void **)&D_spatialFilter, sizeof(float) * iSpatialFilterSize), "prepareCuFFTFilter malloc")) {
				cudaFree(D_filter);
				D_filter = nullptr;
				return false;
			}
			if (!checkCuda(cudaMemcpy(D_spatialFilter, &hostSpatialFilter[0], sizeof(float) * iSpatialFilterSize, cudaMemcpyHostToDevice), "prepareCuFFTFilter memcpy")) {
				cudaFree(D_filter);
				D_filter = nullptr;
				return false;
			}

			bool ok = runCudaFFT(filterRows, D_spatialFilter, iPaddedDetCount, iPaddedDetCount, iPaddedDetCount, D_filter, stream());

			// need to synchronize here for the cudaFree
			ok &= stream.sync("prepareCuFFTFilter FFT");

			cudaFree(D_spatialFilter);

			if (!ok) {
				cudaFree(D_filter);
				D_filter = nullptr;
				return false;
			}
			break;
		}
		default:
		{
			ASTRA_ERROR("FBP::setFilter: Unknown filter type requested");
			return false;
		}
	}

	return true;
}

}
