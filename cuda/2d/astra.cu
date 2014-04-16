/*
-----------------------------------------------------------------------
Copyright 2012 iMinds-Vision Lab, University of Antwerp

Contact: astra@ua.ac.be
Website: http://astra.ua.ac.be


This file is part of the
All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA Toolbox").

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

#include <cstdio>
#include <cassert>

#include "util.h"
#include "par_fp.h"
#include "fan_fp.h"
#include "par_bp.h"
#include "fan_bp.h"
#include "arith.h"
#include "astra.h"

#include "fft.h"

#include <fstream>
#include <cuda.h>

#include "../../include/astra/Logger.h"

// For fan beam FBP weighting
#include "../3d/fdk.h"

using namespace astraCUDA;
using namespace std;


namespace astra {

enum CUDAProjectionType {
	PROJ_PARALLEL,
	PROJ_FAN
};


class AstraFBP_internal {
public:
	SDimensions dims;
	float* angles;
	float* TOffsets;

	float fOriginSourceDistance;
	float fOriginDetectorDistance;

	float fPixelSize;

	bool bFanBeam;
	bool bShortScan;

	bool initialized;
	bool setStartReconstruction;

	float* D_sinoData;
	unsigned int sinoPitch;

	float* D_volumeData;
	unsigned int volumePitch;

	cufftComplex * m_pDevFilter;
};

AstraFBP::AstraFBP()
{
	pData = new AstraFBP_internal();

	pData->angles = 0;
	pData->D_sinoData = 0;
	pData->D_volumeData = 0;

	pData->dims.iVolWidth = 0;
	pData->dims.iProjAngles = 0;
	pData->dims.fDetScale = 1.0f;
	pData->dims.iRaysPerDet = 1;
	pData->dims.iRaysPerPixelDim = 1;

	pData->initialized = false;
	pData->setStartReconstruction = false;

	pData->m_pDevFilter = NULL;
}

AstraFBP::~AstraFBP()
{
	delete[] pData->angles;
	pData->angles = 0;

	delete[] pData->TOffsets;
	pData->TOffsets = 0;

	cudaFree(pData->D_sinoData);
	pData->D_sinoData = 0;

	cudaFree(pData->D_volumeData);
	pData->D_volumeData = 0;

	if(pData->m_pDevFilter != NULL)
	{
		freeComplexOnDevice(pData->m_pDevFilter);
		pData->m_pDevFilter = NULL;
	}

	delete pData;
	pData = 0;
}

bool AstraFBP::setReconstructionGeometry(unsigned int iVolWidth,
                                          unsigned int iVolHeight,
                                          float fPixelSize)
{
	if (pData->initialized)
		return false;

	pData->dims.iVolWidth = iVolWidth;
	pData->dims.iVolHeight = iVolHeight;

	pData->fPixelSize = fPixelSize;

	return (iVolWidth > 0 && iVolHeight > 0 && fPixelSize > 0.0f);
}

bool AstraFBP::setProjectionGeometry(unsigned int iProjAngles,
                                      unsigned int iProjDets,
                                      const float* pfAngles,
                                      float fDetSize)
{
	if (pData->initialized)
		return false;

	pData->dims.iProjAngles = iProjAngles;
	pData->dims.iProjDets = iProjDets;
	pData->dims.fDetScale = fDetSize / pData->fPixelSize;

	if (iProjAngles == 0 || iProjDets == 0 || pfAngles == 0)
		return false;

	pData->angles = new float[iProjAngles];
	memcpy(pData->angles, pfAngles, iProjAngles * sizeof(pfAngles[0]));

	pData->bFanBeam = false;

	return true;
}

bool AstraFBP::setFanGeometry(unsigned int iProjAngles,
                              unsigned int iProjDets,
                              const float* pfAngles,
                              float fOriginSourceDistance,
                              float fOriginDetectorDistance,
                              float fDetSize,
                              bool bShortScan)
{
	// Slightly abusing setProjectionGeometry for this...
	if (!setProjectionGeometry(iProjAngles, iProjDets, pfAngles, fDetSize))
		return false;

	pData->fOriginSourceDistance = fOriginSourceDistance;
	pData->fOriginDetectorDistance = fOriginDetectorDistance;

	pData->bFanBeam = true;
	pData->bShortScan = bShortScan;

	return true;
}


bool AstraFBP::setPixelSuperSampling(unsigned int iPixelSuperSampling)
{
	if (pData->initialized)
		return false;

	if (iPixelSuperSampling == 0)
		return false;

	pData->dims.iRaysPerPixelDim = iPixelSuperSampling;

	return true;
}


bool AstraFBP::setTOffsets(const float* pfTOffsets)
{
	if (pData->initialized)
		return false;

	if (pfTOffsets == 0)
		return false;

	pData->TOffsets = new float[pData->dims.iProjAngles];
	memcpy(pData->TOffsets, pfTOffsets, pData->dims.iProjAngles * sizeof(pfTOffsets[0]));

	return true;
}

bool AstraFBP::init(int iGPUIndex)
{
	if (pData->initialized)
	{
		return false;
	}

	if (pData->dims.iProjAngles == 0 || pData->dims.iVolWidth == 0)
	{
		return false;
	}

	if (iGPUIndex != -1) {
		cudaSetDevice(iGPUIndex);
		cudaError_t err = cudaGetLastError();

		// Ignore errors caused by calling cudaSetDevice multiple times
		if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
		{
			return false;
		}
	}

	bool ok = allocateVolumeData(pData->D_volumeData, pData->volumePitch, pData->dims);
	if (!ok)
	{
		return false;
	}

	ok = allocateProjectionData(pData->D_sinoData, pData->sinoPitch, pData->dims);
	if (!ok)
	{
		cudaFree(pData->D_volumeData);
		pData->D_volumeData = 0;
		return false;
	}

	pData->initialized = true;

	return true;
}

bool AstraFBP::setSinogram(const float* pfSinogram,
                            unsigned int iSinogramPitch)
{
	if (!pData->initialized)
		return false;
	if (!pfSinogram)
		return false;

	bool ok = copySinogramToDevice(pfSinogram, iSinogramPitch,
	                               pData->dims.iProjDets,
	                               pData->dims.iProjAngles,
	                               pData->D_sinoData, pData->sinoPitch);
	if (!ok)
		return false;

	// rescale sinogram to adjust for pixel size
	processVol<opMul>(pData->D_sinoData,
	                       1.0f/(pData->fPixelSize*pData->fPixelSize),
	                       pData->sinoPitch,
	                       pData->dims.iProjDets, pData->dims.iProjAngles);

	pData->setStartReconstruction = false;

	return true;
}

static int calcNextPowerOfTwo(int _iValue)
{
	int iOutput = 1;

	while(iOutput < _iValue)
	{
		iOutput *= 2;
	}

	return iOutput;
}

bool AstraFBP::run()
{
	if (!pData->initialized)
	{
		return false;
	}

	zeroVolumeData(pData->D_volumeData, pData->volumePitch, pData->dims);

	bool ok = false;

	if (pData->bFanBeam) {
		// Call FDK_PreWeight to handle fan beam geometry. We treat
		// this as a cone beam setup of a single slice:

		// TODO: TOffsets...

		// We create a fake cudaPitchedPtr
		cudaPitchedPtr tmp;
		tmp.ptr = pData->D_sinoData;
		tmp.pitch = pData->sinoPitch * sizeof(float);
		tmp.xsize = pData->dims.iProjDets;
		tmp.ysize = pData->dims.iProjAngles;
		// and a fake Dimensions3D
		astraCUDA3d::SDimensions3D dims3d;
		dims3d.iVolX = pData->dims.iVolWidth;
		dims3d.iVolY = pData->dims.iVolHeight;
		dims3d.iVolZ = 1;
		dims3d.iProjAngles = pData->dims.iProjAngles;
		dims3d.iProjU = pData->dims.iProjDets;
		dims3d.iProjV = 1;
		dims3d.iRaysPerDetDim = dims3d.iRaysPerVoxelDim = 1;

		astraCUDA3d::FDK_PreWeight(tmp, pData->fOriginSourceDistance,
		              pData->fOriginDetectorDistance, 0.0f, 0.0f,
		              pData->dims.fDetScale, 1.0f, // TODO: Are these correct?
		              pData->bShortScan, dims3d, pData->angles);
	}

	if (pData->m_pDevFilter) {

		int iFFTRealDetCount = calcNextPowerOfTwo(2 * pData->dims.iProjDets);
		int iFFTFourDetCount = calcFFTFourSize(iFFTRealDetCount);

		cufftComplex * pDevComplexSinogram = NULL;

		allocateComplexOnDevice(pData->dims.iProjAngles, iFFTFourDetCount, &pDevComplexSinogram);

		runCudaFFT(pData->dims.iProjAngles, pData->D_sinoData, pData->sinoPitch, pData->dims.iProjDets, iFFTRealDetCount, iFFTFourDetCount, pDevComplexSinogram);

		applyFilter(pData->dims.iProjAngles, iFFTFourDetCount, pDevComplexSinogram, pData->m_pDevFilter);

		runCudaIFFT(pData->dims.iProjAngles, pDevComplexSinogram, pData->D_sinoData, pData->sinoPitch, pData->dims.iProjDets, iFFTRealDetCount, iFFTFourDetCount);

		freeComplexOnDevice(pDevComplexSinogram);

	}

	if (pData->bFanBeam) {
		// TODO: TOffsets?
		// TODO: Remove this code duplication with CudaReconstructionAlgorithm
		SFanProjection* projs;
		projs = new SFanProjection[pData->dims.iProjAngles];

		float fSrcX0 = 0.0f;
		float fSrcY0 = -pData->fOriginSourceDistance / pData->fPixelSize;
		float fDetUX0 = pData->dims.fDetScale;
		float fDetUY0 = 0.0f;
		float fDetSX0 = pData->dims.iProjDets * fDetUX0 / -2.0f;
		float fDetSY0 = pData->fOriginDetectorDistance / pData->fPixelSize;

#define ROTATE0(name,i,alpha) do { projs[i].f##name##X = f##name##X0 * cos(alpha) - f##name##Y0 * sin(alpha); projs[i].f##name##Y = f##name##X0 * sin(alpha) + f##name##Y0 * cos(alpha); } while(0)
		for (unsigned int i = 0; i < pData->dims.iProjAngles; ++i) {
			ROTATE0(Src, i, pData->angles[i]);
			ROTATE0(DetS, i, pData->angles[i]);
			ROTATE0(DetU, i, pData->angles[i]);
		}

#undef ROTATE0
		ok = FanBP_FBPWeighted(pData->D_volumeData, pData->volumePitch, pData->D_sinoData, pData->sinoPitch, pData->dims, projs);

		delete[] projs;

	} else {
		ok = BP(pData->D_volumeData, pData->volumePitch, pData->D_sinoData, pData->sinoPitch, pData->dims, pData->angles, pData->TOffsets);
	}
	if(!ok)
	{
		return false;
	}

	processVol<opMul>(pData->D_volumeData,
	                      (M_PI / 2.0f) / (float)pData->dims.iProjAngles,
	                      pData->volumePitch,
	                      pData->dims.iVolWidth, pData->dims.iVolHeight);

	return true;
}

bool AstraFBP::getReconstruction(float* pfReconstruction, unsigned int iReconstructionPitch) const
{
	if (!pData->initialized)
		return false;

	bool ok = copyVolumeFromDevice(pfReconstruction, iReconstructionPitch,
	                               pData->dims.iVolWidth,
	                               pData->dims.iVolHeight,
	                               pData->D_volumeData, pData->volumePitch);
	if (!ok)
		return false;

	return true;
}

int AstraFBP::calcFourierFilterSize(int _iDetectorCount)
{
	int iFFTRealDetCount = calcNextPowerOfTwo(2 * _iDetectorCount);
	int iFreqBinCount = calcFFTFourSize(iFFTRealDetCount);

	// CHECKME: Matlab makes this at least 64. Do we also need to?
	return iFreqBinCount;
}

bool AstraFBP::setFilter(E_FBPFILTER _eFilter, const float * _pfHostFilter /* = NULL */, int _iFilterWidth /* = 0 */, float _fD /* = 1.0f */, float _fFilterParameter /* = -1.0f */)
{
	if(pData->m_pDevFilter != 0)
	{
		freeComplexOnDevice(pData->m_pDevFilter);
		pData->m_pDevFilter = 0;
	}

	if (_eFilter == FILTER_NONE)
		return true; // leave pData->m_pDevFilter set to 0


	int iFFTRealDetCount = calcNextPowerOfTwo(2 * pData->dims.iProjDets);
	int iFreqBinCount = calcFFTFourSize(iFFTRealDetCount);

	cufftComplex * pHostFilter = new cufftComplex[pData->dims.iProjAngles * iFreqBinCount];
	memset(pHostFilter, 0, sizeof(cufftComplex) * pData->dims.iProjAngles * iFreqBinCount);

	allocateComplexOnDevice(pData->dims.iProjAngles, iFreqBinCount, &(pData->m_pDevFilter));

	switch(_eFilter)
	{
		case FILTER_NONE:
			// handled above
			break;
		case FILTER_RAMLAK:
		case FILTER_SHEPPLOGAN:
		case FILTER_COSINE:
		case FILTER_HAMMING:
		case FILTER_HANN:
		case FILTER_TUKEY:
		case FILTER_LANCZOS:
		case FILTER_TRIANGULAR:
		case FILTER_GAUSSIAN:
		case FILTER_BARTLETTHANN:
		case FILTER_BLACKMAN:
		case FILTER_NUTTALL:
		case FILTER_BLACKMANHARRIS:
		case FILTER_BLACKMANNUTTALL:
		case FILTER_FLATTOP:
		case FILTER_PARZEN:
		{
			genFilter(_eFilter, _fD, pData->dims.iProjAngles, pHostFilter, iFFTRealDetCount, iFreqBinCount, _fFilterParameter);
			uploadComplexArrayToDevice(pData->dims.iProjAngles, iFreqBinCount, pHostFilter, pData->m_pDevFilter);

			break;
		}
		case FILTER_PROJECTION:
		{
			// make sure the offered filter has the correct size
			assert(_iFilterWidth == iFreqBinCount);

			for(int iFreqBinIndex = 0; iFreqBinIndex < iFreqBinCount; iFreqBinIndex++)
			{
				float fValue = _pfHostFilter[iFreqBinIndex];

				for(int iProjectionIndex = 0; iProjectionIndex < (int)pData->dims.iProjAngles; iProjectionIndex++)
				{
					pHostFilter[iFreqBinIndex + iProjectionIndex * iFreqBinCount].x = fValue;
					pHostFilter[iFreqBinIndex + iProjectionIndex * iFreqBinCount].y = 0.0f;
				}
			}
			uploadComplexArrayToDevice(pData->dims.iProjAngles, iFreqBinCount, pHostFilter, pData->m_pDevFilter);
			break;
		}
		case FILTER_SINOGRAM:
		{
			// make sure the offered filter has the correct size
			assert(_iFilterWidth == iFreqBinCount);

			for(int iFreqBinIndex = 0; iFreqBinIndex < iFreqBinCount; iFreqBinIndex++)
			{
				for(int iProjectionIndex = 0; iProjectionIndex < (int)pData->dims.iProjAngles; iProjectionIndex++)
				{
					float fValue = _pfHostFilter[iFreqBinIndex + iProjectionIndex * _iFilterWidth];

					pHostFilter[iFreqBinIndex + iProjectionIndex * iFreqBinCount].x = fValue;
					pHostFilter[iFreqBinIndex + iProjectionIndex * iFreqBinCount].y = 0.0f;
				}
			}
			uploadComplexArrayToDevice(pData->dims.iProjAngles, iFreqBinCount, pHostFilter, pData->m_pDevFilter);
			break;
		}
		case FILTER_RPROJECTION:
		{
			int iProjectionCount = pData->dims.iProjAngles;
			int iRealFilterElementCount = iProjectionCount * iFFTRealDetCount;
			float * pfHostRealFilter = new float[iRealFilterElementCount];
			memset(pfHostRealFilter, 0, sizeof(float) * iRealFilterElementCount);

			int iUsedFilterWidth = min(_iFilterWidth, iFFTRealDetCount);
			int iStartFilterIndex = (_iFilterWidth - iUsedFilterWidth) / 2;
			int iMaxFilterIndex = iStartFilterIndex + iUsedFilterWidth;

			int iFilterShiftSize = _iFilterWidth / 2;

			for(int iDetectorIndex = iStartFilterIndex; iDetectorIndex < iMaxFilterIndex; iDetectorIndex++)
			{
				int iFFTInFilterIndex = (iDetectorIndex + iFFTRealDetCount - iFilterShiftSize) % iFFTRealDetCount;
				float fValue = _pfHostFilter[iDetectorIndex];

				for(int iProjectionIndex = 0; iProjectionIndex < (int)pData->dims.iProjAngles; iProjectionIndex++)
				{
					pfHostRealFilter[iFFTInFilterIndex + iProjectionIndex * iFFTRealDetCount] = fValue;
				}
			}

			float* pfDevRealFilter = NULL;
			cudaMalloc((void **)&pfDevRealFilter, sizeof(float) * iRealFilterElementCount); // TODO: check for errors
			cudaMemcpy(pfDevRealFilter, pfHostRealFilter, sizeof(float) * iRealFilterElementCount, cudaMemcpyHostToDevice);
			delete[] pfHostRealFilter;

			runCudaFFT(iProjectionCount, pfDevRealFilter, iFFTRealDetCount, iFFTRealDetCount, iFFTRealDetCount, iFreqBinCount, pData->m_pDevFilter);

			cudaFree(pfDevRealFilter);

			break;
		}
		case FILTER_RSINOGRAM:
		{
			int iProjectionCount = pData->dims.iProjAngles;
			int iRealFilterElementCount = iProjectionCount * iFFTRealDetCount;
			float* pfHostRealFilter = new float[iRealFilterElementCount];
			memset(pfHostRealFilter, 0, sizeof(float) * iRealFilterElementCount);

			int iUsedFilterWidth = min(_iFilterWidth, iFFTRealDetCount);
			int iStartFilterIndex = (_iFilterWidth - iUsedFilterWidth) / 2;
			int iMaxFilterIndex = iStartFilterIndex + iUsedFilterWidth;

			int iFilterShiftSize = _iFilterWidth / 2;
			
			for(int iDetectorIndex = iStartFilterIndex; iDetectorIndex < iMaxFilterIndex; iDetectorIndex++)
			{
				int iFFTInFilterIndex = (iDetectorIndex + iFFTRealDetCount - iFilterShiftSize) % iFFTRealDetCount;

				for(int iProjectionIndex = 0; iProjectionIndex < (int)pData->dims.iProjAngles; iProjectionIndex++)
				{
					float fValue = _pfHostFilter[iDetectorIndex + iProjectionIndex * _iFilterWidth];
					pfHostRealFilter[iFFTInFilterIndex + iProjectionIndex * iFFTRealDetCount] = fValue;
				}
			}

			float* pfDevRealFilter = NULL;
			cudaMalloc((void **)&pfDevRealFilter, sizeof(float) * iRealFilterElementCount); // TODO: check for errors
			cudaMemcpy(pfDevRealFilter, pfHostRealFilter, sizeof(float) * iRealFilterElementCount, cudaMemcpyHostToDevice);
			delete[] pfHostRealFilter;

			runCudaFFT(iProjectionCount, pfDevRealFilter, iFFTRealDetCount, iFFTRealDetCount, iFFTRealDetCount, iFreqBinCount, pData->m_pDevFilter);

			cudaFree(pfDevRealFilter);

			break;
		}
		default:
		{
			fprintf(stderr, "AstraFBP::setFilter: Unknown filter type requested");
			delete [] pHostFilter;
			return false;
		}
	}

	delete [] pHostFilter;

	return true;
}

BPalgo::BPalgo()
{

}

BPalgo::~BPalgo()
{

}

bool BPalgo::init()
{
	return true;
}

bool BPalgo::iterate(unsigned int)
{
	// TODO: This zeroVolume makes an earlier memcpy of D_volumeData redundant
	zeroVolumeData(D_volumeData, volumePitch, dims);
	callBP(D_volumeData, volumePitch, D_sinoData, sinoPitch);
	return true;
}

float BPalgo::computeDiffNorm()
{
	float *D_projData;
	unsigned int projPitch;

	allocateProjectionData(D_projData, projPitch, dims);

	cudaMemcpy2D(D_projData, sizeof(float)*projPitch, D_sinoData, sizeof(float)*sinoPitch, sizeof(float)*dims.iProjDets, dims.iProjAngles, cudaMemcpyDeviceToDevice);
	callFP(D_volumeData, volumePitch, D_projData, projPitch, -1.0f);

	float s = dotProduct2D(D_projData, projPitch, dims.iProjDets, dims.iProjAngles);

	cudaFree(D_projData);

	return sqrt(s);
}


bool astraCudaFP(const float* pfVolume, float* pfSinogram,
                 unsigned int iVolWidth, unsigned int iVolHeight,
                 unsigned int iProjAngles, unsigned int iProjDets,
                 const float *pfAngles, const float *pfOffsets,
                 float fDetSize, unsigned int iDetSuperSampling,
                 int iGPUIndex)
{
	SDimensions dims;

	if (iProjAngles == 0 || iProjDets == 0 || pfAngles == 0)
		return false;

	dims.iProjAngles = iProjAngles;
	dims.iProjDets = iProjDets;
	dims.fDetScale = fDetSize;

	if (iDetSuperSampling == 0)
		return false;

	dims.iRaysPerDet = iDetSuperSampling;

	if (iVolWidth <= 0 || iVolHeight <= 0)
		return false;

	dims.iVolWidth = iVolWidth;
	dims.iVolHeight = iVolHeight;

	if (iGPUIndex != -1) {
		cudaSetDevice(iGPUIndex);
		cudaError_t err = cudaGetLastError();

		// Ignore errors caused by calling cudaSetDevice multiple times
		if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
			return false;
	}

	bool ok;

	float* D_volumeData;
	unsigned int volumePitch;

	ok = allocateVolumeData(D_volumeData, volumePitch, dims);
	if (!ok)
		return false;

	float* D_sinoData;
	unsigned int sinoPitch;

	ok = allocateProjectionData(D_sinoData, sinoPitch, dims);
	if (!ok) {
		cudaFree(D_volumeData);
		return false;
	}

	ok = copyVolumeToDevice(pfVolume, dims.iVolWidth,
	                        dims.iVolWidth, dims.iVolHeight,
	                        D_volumeData, volumePitch);
	if (!ok) {
		cudaFree(D_volumeData);
		cudaFree(D_sinoData);
		return false;
	}

	zeroProjectionData(D_sinoData, sinoPitch, dims);
	ok = FP(D_volumeData, volumePitch, D_sinoData, sinoPitch, dims, pfAngles, pfOffsets, 1.0f);
	if (!ok) {
		cudaFree(D_volumeData);
		cudaFree(D_sinoData);
		return false;
	}

	ok = copySinogramFromDevice(pfSinogram, dims.iProjDets,
	                            dims.iProjDets,
	                            dims.iProjAngles,
	                            D_sinoData, sinoPitch);
	if (!ok) {
		cudaFree(D_volumeData);
		cudaFree(D_sinoData);
		return false;
	}

	cudaFree(D_volumeData);
	cudaFree(D_sinoData);
	return true;
}

bool astraCudaFanFP(const float* pfVolume, float* pfSinogram,
                    unsigned int iVolWidth, unsigned int iVolHeight,
                    unsigned int iProjAngles, unsigned int iProjDets,
                    const float *pfAngles, float fOriginSourceDistance,
                    float fOriginDetectorDistance, float fPixelSize,
                    float fDetSize,
                    unsigned int iDetSuperSampling,
                    int iGPUIndex)
{
	SDimensions dims;

	if (iProjAngles == 0 || iProjDets == 0 || pfAngles == 0)
		return false;

	dims.iProjAngles = iProjAngles;
	dims.iProjDets = iProjDets;

	if (iDetSuperSampling == 0)
		return false;

	dims.iRaysPerDet = iDetSuperSampling;

	if (iVolWidth <= 0 || iVolHeight <= 0)
		return false;

	dims.iVolWidth = iVolWidth;
	dims.iVolHeight = iVolHeight;

	if (iGPUIndex != -1) {
		cudaSetDevice(iGPUIndex);
		cudaError_t err = cudaGetLastError();

		// Ignore errors caused by calling cudaSetDevice multiple times
		if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
			return false;
	}

	bool ok;

	float* D_volumeData;
	unsigned int volumePitch;

	ok = allocateVolumeData(D_volumeData, volumePitch, dims);
	if (!ok)
		return false;

	float* D_sinoData;
	unsigned int sinoPitch;

	ok = allocateProjectionData(D_sinoData, sinoPitch, dims);
	if (!ok) {
		cudaFree(D_volumeData);
		return false;
	}

	ok = copyVolumeToDevice(pfVolume, dims.iVolWidth,
	                        dims.iVolWidth, dims.iVolHeight,
	                        D_volumeData, volumePitch);
	if (!ok) {
		cudaFree(D_volumeData);
		cudaFree(D_sinoData);
		return false;
	}

	zeroProjectionData(D_sinoData, sinoPitch, dims);

	// TODO: Turn this geometry conversion into a util function
	SFanProjection* projs = new SFanProjection[dims.iProjAngles];

	float fSrcX0 = 0.0f;
	float fSrcY0 = -fOriginSourceDistance / fPixelSize;
	float fDetUX0 = fDetSize / fPixelSize;
	float fDetUY0 = 0.0f;
	float fDetSX0 = dims.iProjDets * fDetUX0 / -2.0f;
	float fDetSY0 = fOriginDetectorDistance / fPixelSize;

#define ROTATE0(name,i,alpha) do { projs[i].f##name##X = f##name##X0 * cos(alpha) - f##name##Y0 * sin(alpha); projs[i].f##name##Y = f##name##X0 * sin(alpha) + f##name##Y0 * cos(alpha); } while(0)
	for (int i = 0; i < dims.iProjAngles; ++i) {
		ROTATE0(Src, i, pfAngles[i]);
		ROTATE0(DetS, i, pfAngles[i]);
		ROTATE0(DetU, i, pfAngles[i]);
	}

#undef ROTATE0

	ok = FanFP(D_volumeData, volumePitch, D_sinoData, sinoPitch, dims, projs, 1.0f);
	delete[] projs;

	if (!ok) {
		cudaFree(D_volumeData);
		cudaFree(D_sinoData);
		return false;
	}

	ok = copySinogramFromDevice(pfSinogram, dims.iProjDets,
	                            dims.iProjDets,
	                            dims.iProjAngles,
	                            D_sinoData, sinoPitch);
	if (!ok) {
		cudaFree(D_volumeData);
		cudaFree(D_sinoData);
		return false;
	}

	cudaFree(D_volumeData);
	cudaFree(D_sinoData);

	return true;

}


bool astraCudaFanFP(const float* pfVolume, float* pfSinogram,
                    unsigned int iVolWidth, unsigned int iVolHeight,
                    unsigned int iProjAngles, unsigned int iProjDets,
                    const SFanProjection *pAngles,
                    unsigned int iDetSuperSampling,
                    int iGPUIndex)
{
	SDimensions dims;

	if (iProjAngles == 0 || iProjDets == 0 || pAngles == 0)
		return false;

	dims.iProjAngles = iProjAngles;
	dims.iProjDets = iProjDets;
	dims.fDetScale = 1.0f; // TODO?

	if (iDetSuperSampling == 0)
		return false;

	dims.iRaysPerDet = iDetSuperSampling;

	if (iVolWidth <= 0 || iVolHeight <= 0)
		return false;

	dims.iVolWidth = iVolWidth;
	dims.iVolHeight = iVolHeight;

	if (iGPUIndex != -1) {
		cudaSetDevice(iGPUIndex);
		cudaError_t err = cudaGetLastError();

		// Ignore errors caused by calling cudaSetDevice multiple times
		if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
			return false;
	}

	bool ok;

	float* D_volumeData;
	unsigned int volumePitch;

	ok = allocateVolumeData(D_volumeData, volumePitch, dims);
	if (!ok)
		return false;

	float* D_sinoData;
	unsigned int sinoPitch;

	ok = allocateProjectionData(D_sinoData, sinoPitch, dims);
	if (!ok) {
		cudaFree(D_volumeData);
		return false;
	}

	ok = copyVolumeToDevice(pfVolume, dims.iVolWidth,
	                        dims.iVolWidth, dims.iVolHeight,
	                        D_volumeData, volumePitch);
	if (!ok) {
		cudaFree(D_volumeData);
		cudaFree(D_sinoData);
		return false;
	}

	zeroProjectionData(D_sinoData, sinoPitch, dims);

	ok = FanFP(D_volumeData, volumePitch, D_sinoData, sinoPitch, dims, pAngles, 1.0f);

	if (!ok) {
		cudaFree(D_volumeData);
		cudaFree(D_sinoData);
		return false;
	}

	ok = copySinogramFromDevice(pfSinogram, dims.iProjDets,
	                            dims.iProjDets,
	                            dims.iProjAngles,
	                            D_sinoData, sinoPitch);
	if (!ok) {
		cudaFree(D_volumeData);
		cudaFree(D_sinoData);
		return false;
	}

	cudaFree(D_volumeData);
	cudaFree(D_sinoData);

	return true;

}


}
