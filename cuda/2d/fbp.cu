/*
-----------------------------------------------------------------------
Copyright: 2010-2018, imec Vision Lab, University of Antwerp
           2014-2018, CWI, Amsterdam

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

#include "astra/cuda/2d/fbp.h"
#include "astra/cuda/2d/fft.h"
#include "astra/cuda/2d/par_bp.h"
#include "astra/cuda/2d/fan_bp.h"
#include "astra/cuda/2d/util.h"

// For fan-beam preweighting
#include "astra/cuda/3d/fdk.h"

#include "astra/Logging.h"
#include "astra/Filters.h"

#include <cuda.h>

namespace astraCUDA {


// static
int FBP::calcFourierFilterSize(int _iDetectorCount)
{
	int iFFTRealDetCount = astra::calcNextPowerOfTwo(2 * _iDetectorCount);
	int iFreqBinCount = astra::calcFFTFourierSize(iFFTRealDetCount);

	// CHECKME: Matlab makes this at least 64. Do we also need to?
	return iFreqBinCount;
}




FBP::FBP() : ReconAlgo()
{
	D_filter = 0;

}

FBP::~FBP()
{
	reset();
}

void FBP::reset()
{
	if (D_filter) {
		freeComplexOnDevice((cufftComplex *)D_filter);
		D_filter = 0;
	}
}

bool FBP::init()
{
	return true;
}

bool FBP::setFilter(const astra::SFilterConfig &_cfg)
{
	if (D_filter)
	{
		freeComplexOnDevice((cufftComplex*)D_filter);
		D_filter = 0;
	}

	if (_cfg.m_eType == astra::FILTER_NONE)
		return true; // leave D_filter set to 0


	int iFFTRealDetCount = astra::calcNextPowerOfTwo(2 * dims.iProjDets);
	int iFreqBinCount = astra::calcFFTFourierSize(iFFTRealDetCount);

	cufftComplex * pHostFilter = new cufftComplex[dims.iProjAngles * iFreqBinCount];
	memset(pHostFilter, 0, sizeof(cufftComplex) * dims.iProjAngles * iFreqBinCount);

	allocateComplexOnDevice(dims.iProjAngles, iFreqBinCount, (cufftComplex**)&D_filter);

	switch(_cfg.m_eType)
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
		case astra::FILTER_PARZEN:
		{
			genCuFFTFilter(_cfg, dims.iProjAngles, pHostFilter, iFFTRealDetCount, iFreqBinCount);
			uploadComplexArrayToDevice(dims.iProjAngles, iFreqBinCount, pHostFilter, (cufftComplex*)D_filter);

			break;
		}
		case astra::FILTER_PROJECTION:
		{
			// make sure the offered filter has the correct size
			assert(_cfg.m_iCustomFilterWidth == iFreqBinCount);

			for(int iFreqBinIndex = 0; iFreqBinIndex < iFreqBinCount; iFreqBinIndex++)
			{
				float fValue = _cfg.m_pfCustomFilter[iFreqBinIndex];

				for(int iProjectionIndex = 0; iProjectionIndex < (int)dims.iProjAngles; iProjectionIndex++)
				{
					pHostFilter[iFreqBinIndex + iProjectionIndex * iFreqBinCount].x = fValue;
					pHostFilter[iFreqBinIndex + iProjectionIndex * iFreqBinCount].y = 0.0f;
				}
			}
			uploadComplexArrayToDevice(dims.iProjAngles, iFreqBinCount, pHostFilter, (cufftComplex*)D_filter);
			break;
		}
		case astra::FILTER_SINOGRAM:
		{
			// make sure the offered filter has the correct size
			assert(_cfg.m_iCustomFilterWidth == iFreqBinCount);

			for(int iFreqBinIndex = 0; iFreqBinIndex < iFreqBinCount; iFreqBinIndex++)
			{
				for(int iProjectionIndex = 0; iProjectionIndex < (int)dims.iProjAngles; iProjectionIndex++)
				{
					float fValue = _cfg.m_pfCustomFilter[iFreqBinIndex + iProjectionIndex * _cfg.m_iCustomFilterWidth];

					pHostFilter[iFreqBinIndex + iProjectionIndex * iFreqBinCount].x = fValue;
					pHostFilter[iFreqBinIndex + iProjectionIndex * iFreqBinCount].y = 0.0f;
				}
			}
			uploadComplexArrayToDevice(dims.iProjAngles, iFreqBinCount, pHostFilter, (cufftComplex*)D_filter);
			break;
		}
		case astra::FILTER_RPROJECTION:
		{
			int iProjectionCount = dims.iProjAngles;
			int iRealFilterElementCount = iProjectionCount * iFFTRealDetCount;
			float * pfHostRealFilter = new float[iRealFilterElementCount];
			memset(pfHostRealFilter, 0, sizeof(float) * iRealFilterElementCount);

			int iUsedFilterWidth = min(_cfg.m_iCustomFilterWidth, iFFTRealDetCount);
			int iStartFilterIndex = (_cfg.m_iCustomFilterWidth - iUsedFilterWidth) / 2;
			int iMaxFilterIndex = iStartFilterIndex + iUsedFilterWidth;

			int iFilterShiftSize = _cfg.m_iCustomFilterWidth / 2;

			for(int iDetectorIndex = iStartFilterIndex; iDetectorIndex < iMaxFilterIndex; iDetectorIndex++)
			{
				int iFFTInFilterIndex = (iDetectorIndex + iFFTRealDetCount - iFilterShiftSize) % iFFTRealDetCount;
				float fValue = _cfg.m_pfCustomFilter[iDetectorIndex];

				for(int iProjectionIndex = 0; iProjectionIndex < (int)dims.iProjAngles; iProjectionIndex++)
				{
					pfHostRealFilter[iFFTInFilterIndex + iProjectionIndex * iFFTRealDetCount] = fValue;
				}
			}

			float* pfDevRealFilter = NULL;
			cudaMalloc((void **)&pfDevRealFilter, sizeof(float) * iRealFilterElementCount); // TODO: check for errors
			cudaMemcpy(pfDevRealFilter, pfHostRealFilter, sizeof(float) * iRealFilterElementCount, cudaMemcpyHostToDevice);
			delete[] pfHostRealFilter;

			runCudaFFT(iProjectionCount, pfDevRealFilter, iFFTRealDetCount, iFFTRealDetCount, iFFTRealDetCount, iFreqBinCount, (cufftComplex*)D_filter);

			cudaFree(pfDevRealFilter);

			break;
		}
		case astra::FILTER_RSINOGRAM:
		{
			int iProjectionCount = dims.iProjAngles;
			int iRealFilterElementCount = iProjectionCount * iFFTRealDetCount;
			float* pfHostRealFilter = new float[iRealFilterElementCount];
			memset(pfHostRealFilter, 0, sizeof(float) * iRealFilterElementCount);

			int iUsedFilterWidth = min(_cfg.m_iCustomFilterWidth, iFFTRealDetCount);
			int iStartFilterIndex = (_cfg.m_iCustomFilterWidth - iUsedFilterWidth) / 2;
			int iMaxFilterIndex = iStartFilterIndex + iUsedFilterWidth;

			int iFilterShiftSize = _cfg.m_iCustomFilterWidth / 2;

			for(int iDetectorIndex = iStartFilterIndex; iDetectorIndex < iMaxFilterIndex; iDetectorIndex++)
			{
				int iFFTInFilterIndex = (iDetectorIndex + iFFTRealDetCount - iFilterShiftSize) % iFFTRealDetCount;

				for(int iProjectionIndex = 0; iProjectionIndex < (int)dims.iProjAngles; iProjectionIndex++)
				{
					float fValue = _cfg.m_pfCustomFilter[iDetectorIndex + iProjectionIndex * _cfg.m_iCustomFilterWidth];
					pfHostRealFilter[iFFTInFilterIndex + iProjectionIndex * iFFTRealDetCount] = fValue;
				}
			}

			float* pfDevRealFilter = NULL;
			cudaMalloc((void **)&pfDevRealFilter, sizeof(float) * iRealFilterElementCount); // TODO: check for errors
			cudaMemcpy(pfDevRealFilter, pfHostRealFilter, sizeof(float) * iRealFilterElementCount, cudaMemcpyHostToDevice);
			delete[] pfHostRealFilter;

			runCudaFFT(iProjectionCount, pfDevRealFilter, iFFTRealDetCount, iFFTRealDetCount, iFFTRealDetCount, iFreqBinCount, (cufftComplex*)D_filter);

			cudaFree(pfDevRealFilter);

			break;
		}
		default:
		{
			ASTRA_ERROR("FBP::setFilter: Unknown filter type requested");
			delete [] pHostFilter;
			return false;
		}
	}

	delete [] pHostFilter;

	return true;
}

bool FBP::iterate(unsigned int iterations)
{
	zeroVolumeData(D_volumeData, volumePitch, dims);

	bool ok = false;

	float fFanDetSize = 0.0f;
	if (fanProjs) {
		// Call FDK_PreWeight to handle fan beam geometry. We treat
		// this as a cone beam setup of a single slice:

		// TODO: TOffsets affects this preweighting...

		// TODO: We take the fan parameters from the last projection here
		// without checking if they're the same in all projections

		float *pfAngles = new float[dims.iProjAngles];

		float fOriginSource, fOriginDetector, fOffset;
		for (unsigned int i = 0; i < dims.iProjAngles; ++i) {
			bool ok = astra::getFanParameters(fanProjs[i], dims.iProjDets,
			                                  pfAngles[i],
			                                  fOriginSource, fOriginDetector,
			                                  fFanDetSize, fOffset);
			if (!ok) {
				ASTRA_ERROR("FBP_CUDA: Failed to extract circular fan beam parameters from fan beam geometry");
				return false;
			}
		}

		// We create a fake cudaPitchedPtr
		cudaPitchedPtr tmp;
		tmp.ptr = D_sinoData;
		tmp.pitch = sinoPitch * sizeof(float);
		tmp.xsize = dims.iProjDets;
		tmp.ysize = dims.iProjAngles;
		// and a fake Dimensions3D
		astraCUDA3d::SDimensions3D dims3d;
		dims3d.iVolX = dims.iVolWidth;
		dims3d.iVolY = dims.iVolHeight;
		dims3d.iVolZ = 1;
		dims3d.iProjAngles = dims.iProjAngles;
		dims3d.iProjU = dims.iProjDets;
		dims3d.iProjV = 1;

		astraCUDA3d::FDK_PreWeight(tmp, fOriginSource,
		              fOriginDetector, 0.0f,
		              fFanDetSize, 1.0f, /* fPixelSize */ 1.0f,
		              m_bShortScan, dims3d, pfAngles);
	} else {
		// TODO: How should different detector pixel size in different
		// projections be handled?
	}

	if (D_filter) {

		int iFFTRealDetCount = astra::calcNextPowerOfTwo(2 * dims.iProjDets);
		int iFFTFourDetCount = astra::calcFFTFourierSize(iFFTRealDetCount);

		cufftComplex * pDevComplexSinogram = NULL;

		allocateComplexOnDevice(dims.iProjAngles, iFFTFourDetCount, &pDevComplexSinogram);

		runCudaFFT(dims.iProjAngles, D_sinoData, sinoPitch, dims.iProjDets, iFFTRealDetCount, iFFTFourDetCount, pDevComplexSinogram);

		applyFilter(dims.iProjAngles, iFFTFourDetCount, pDevComplexSinogram, (cufftComplex*)D_filter);

		runCudaIFFT(dims.iProjAngles, pDevComplexSinogram, D_sinoData, sinoPitch, dims.iProjDets, iFFTRealDetCount, iFFTFourDetCount);

		freeComplexOnDevice(pDevComplexSinogram);

	}

	if (fanProjs) {
		float fOutputScale = 1.0 / (/*fPixelSize * fPixelSize * fPixelSize * */ fFanDetSize * fFanDetSize);

		ok = FanBP_FBPWeighted(D_volumeData, volumePitch, D_sinoData, sinoPitch, dims, fanProjs, fOutputScale);

	} else {
		// scale by number of angles. For the fan-beam case, this is already
		// handled by FDK_PreWeight
		float fOutputScale = (M_PI / 2.0f) / (float)dims.iProjAngles;
		//fOutputScale /= fDetSize * fDetSize;

		ok = BP(D_volumeData, volumePitch, D_sinoData, sinoPitch, dims, parProjs, fOutputScale);
	}
	if(!ok)
	{
		return false;
	}

	return true;
}


}
