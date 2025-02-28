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

#include "astra/cuda/2d/fbp.h"
#include "astra/cuda/2d/fft.h"
#include "astra/cuda/2d/par_bp.h"
#include "astra/cuda/2d/fan_bp.h"
#include "astra/cuda/2d/util.h"

// For fan-beam preweighting
#include "astra/cuda/3d/fdk.h"

#include "astra/Logging.h"
#include "astra/Filters.h"

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
	m_bShortScan = false;
	fReconstructionScale = 1.0f;
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
	m_bShortScan = false;
	fReconstructionScale = 1.0f;
}

bool FBP::init()
{
	return true;
}

bool FBP::setReconstructionScale(float fScale)
{
	fReconstructionScale = fScale;
	return true;
}

bool FBP::setFilter(const astra::SFilterConfig &_cfg)
{
	if (D_filter)
	{
		freeComplexOnDevice((cufftComplex*)D_filter);
		D_filter = 0;
	}

	cufftComplex *f;
	bool ok = prepareCuFFTFilter(_cfg, f, m_bSingleFilter, dims.iProjAngles, dims.iProjDets);
	D_filter = (void *)f;
	return ok;
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
		              fFanDetSize, 1.0f,
		              m_bShortScan, dims3d, pfAngles);
	} else {
		// TODO: How should different detector pixel size in different
		// projections be handled?
	}

	if (D_filter) {

		int iPaddedSize = astra::calcNextPowerOfTwo(2 * dims.iProjDets);
		int iFourierSize = astra::calcFFTFourierSize(iPaddedSize);

		cufftComplex * D_pcFourier = NULL;

		allocateComplexOnDevice(dims.iProjAngles, iFourierSize, &D_pcFourier);

		runCudaFFT(dims.iProjAngles, D_sinoData, sinoPitch, dims.iProjDets, iPaddedSize, D_pcFourier);

		applyFilter(dims.iProjAngles, iFourierSize, D_pcFourier, (cufftComplex*)D_filter, m_bSingleFilter);

		runCudaIFFT(dims.iProjAngles, D_pcFourier, D_sinoData, sinoPitch, dims.iProjDets, iPaddedSize);

		freeComplexOnDevice(D_pcFourier);

	}

	if (fanProjs) {
		ok = FanBP_FBPWeighted(D_volumeData, volumePitch, D_sinoData, sinoPitch, dims, fanProjs, fProjectorScale * fReconstructionScale);

	} else {
		// scale by number of angles. For the fan-beam case, this is already
		// handled by FDK_PreWeight
		float fOutputScale = (M_PI / 2.0f) / (float)dims.iProjAngles;

		ok = BP(D_volumeData, volumePitch, D_sinoData, sinoPitch, dims, parProjs, fOutputScale * fProjectorScale * fReconstructionScale);
	}
	if(!ok)
	{
		return false;
	}

	return true;
}


}
