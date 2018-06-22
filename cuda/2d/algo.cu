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

#include "astra/cuda/2d/algo.h"
#include "astra/cuda/2d/par_fp.h"
#include "astra/cuda/2d/fan_fp.h"
#include "astra/cuda/2d/par_bp.h"
#include "astra/cuda/2d/fan_bp.h"
#include "astra/cuda/2d/util.h"
#include "astra/cuda/2d/arith.h"
#include "astra/cuda/2d/astra.h"

#include <cassert>

namespace astraCUDA {

ReconAlgo::ReconAlgo()
{
	parProjs = 0;
	fanProjs = 0;
	shouldAbort = false;

	useVolumeMask = false;
	useSinogramMask = false;
	D_maskData = 0;
	D_smaskData = 0;

	D_sinoData = 0;
	D_volumeData = 0;

	useMinConstraint = false;
	useMaxConstraint = false;

	freeGPUMemory = false;
}

ReconAlgo::~ReconAlgo()
{
	reset();
}

void ReconAlgo::reset()
{
	delete[] parProjs;
	delete[] fanProjs;

	if (freeGPUMemory) {
		cudaFree(D_maskData);
		cudaFree(D_smaskData);
		cudaFree(D_sinoData);
		cudaFree(D_volumeData);
	}

	parProjs = 0;
	fanProjs = 0;
	shouldAbort = false;

	useVolumeMask = false;
	useSinogramMask = false;

	D_maskData = 0;
	D_smaskData = 0;

	D_sinoData = 0;
	D_volumeData = 0;
	
	useMinConstraint = false;
	useMaxConstraint = false;

	freeGPUMemory = false;
}

bool ReconAlgo::setGPUIndex(int iGPUIndex)
{
	if (iGPUIndex != -1) {
		cudaSetDevice(iGPUIndex);
		cudaError_t err = cudaGetLastError();

		// Ignore errors caused by calling cudaSetDevice multiple times
		if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
			return false;
	}

	return true;
}

bool ReconAlgo::enableVolumeMask()
{
	useVolumeMask = true;
	return true;
}

bool ReconAlgo::enableSinogramMask()
{
	useSinogramMask = true;
	return true;
}


bool ReconAlgo::setGeometry(const astra::CVolumeGeometry2D* pVolGeom,
                            const astra::CProjectionGeometry2D* pProjGeom)
{
	bool ok;

	ok = convertAstraGeometry_dims(pVolGeom, pProjGeom, dims);

	if (!ok)
		return false;

	delete[] parProjs;
	parProjs = 0;
	delete[] fanProjs;
	fanProjs = 0;

	fOutputScale = 1.0f;
	ok = convertAstraGeometry(pVolGeom, pProjGeom, parProjs, fanProjs, fOutputScale);
	if (!ok)
		return false;

	return true;
}

bool ReconAlgo::setSuperSampling(int raysPerDet, int raysPerPixelDim)
{
	if (raysPerDet <= 0 || raysPerPixelDim <= 0)
		return false;

	dims.iRaysPerDet = raysPerDet;
	dims.iRaysPerPixelDim = raysPerPixelDim;

	return true;
}

bool ReconAlgo::setVolumeMask(float* _D_maskData, unsigned int _maskPitch)
{
	assert(useVolumeMask);

	D_maskData = _D_maskData;
	maskPitch = _maskPitch;

	return true;
}

bool ReconAlgo::setSinogramMask(float* _D_smaskData, unsigned int _smaskPitch)
{
	assert(useSinogramMask);

	D_smaskData = _D_smaskData;
	smaskPitch = _smaskPitch;

	return true;
}

bool ReconAlgo::setBuffers(float* _D_volumeData, unsigned int _volumePitch,
                      float* _D_projData, unsigned int _projPitch)
{
	D_volumeData = _D_volumeData;
	volumePitch = _volumePitch;
	D_sinoData = _D_projData;
	sinoPitch = _projPitch;

	return true;
}

bool ReconAlgo::setMinConstraint(float fMin)
{
	fMinConstraint = fMin;
	useMinConstraint = true;
	return true;
}

bool ReconAlgo::setMaxConstraint(float fMax)
{
	fMaxConstraint = fMax;
	useMaxConstraint = true;
	return true;
}



bool ReconAlgo::allocateBuffers()
{
	bool ok;
	ok = allocateVolumeData(D_volumeData, volumePitch, dims);
	if (!ok)
		return false;

	ok = allocateProjectionData(D_sinoData, sinoPitch, dims);
	if (!ok) {
		cudaFree(D_volumeData);
		D_volumeData = 0;
		return false;
	}

	if (useVolumeMask) {
		ok = allocateVolumeData(D_maskData, maskPitch, dims);
		if (!ok) {
			cudaFree(D_volumeData);
			cudaFree(D_sinoData);
			D_volumeData = 0;
			D_sinoData = 0;
			return false;
		}
	}

	if (useSinogramMask) {
		ok = allocateProjectionData(D_smaskData, smaskPitch, dims);
		if (!ok) {
			cudaFree(D_volumeData);
			cudaFree(D_sinoData);
			cudaFree(D_maskData);
			D_volumeData = 0;
			D_sinoData = 0;
			D_maskData = 0;
			return false;
		}
	}

	freeGPUMemory = true;
	return true;
}

bool ReconAlgo::copyDataToGPU(const float* pfSinogram, unsigned int iSinogramPitch, float fSinogramScale,
                              const float* pfReconstruction, unsigned int iReconstructionPitch,
                              const float* pfVolMask, unsigned int iVolMaskPitch,
                              const float* pfSinoMask, unsigned int iSinoMaskPitch)
{
	if (!pfSinogram)
		return false;
	if (!pfReconstruction)
		return false;

	bool ok = copySinogramToDevice(pfSinogram, iSinogramPitch,
	                               dims,
	                               D_sinoData, sinoPitch);
	if (!ok)
		return false;

	// rescale sinogram to adjust for pixel size
	processSino<opMul>(D_sinoData, fSinogramScale,
	                       //1.0f/(fPixelSize*fPixelSize),
	                       sinoPitch, dims);

	ok = copyVolumeToDevice(pfReconstruction, iReconstructionPitch,
	                        dims,
	                        D_volumeData, volumePitch);
	if (!ok)
		return false;



	if (useVolumeMask) {
		if (!pfVolMask)
			return false;

		ok = copyVolumeToDevice(pfVolMask, iVolMaskPitch,
		                        dims,
		                        D_maskData, maskPitch);
		if (!ok)
			return false;
	}

	if (useSinogramMask) {
		if (!pfSinoMask)
			return false;

		ok = copySinogramToDevice(pfSinoMask, iSinoMaskPitch,
		                          dims,
		                          D_smaskData, smaskPitch);
		if (!ok)
			return false;
	}

	return true;
}

bool ReconAlgo::getReconstruction(float* pfReconstruction,
                                  unsigned int iReconstructionPitch) const
{
	bool ok = copyVolumeFromDevice(pfReconstruction, iReconstructionPitch,
	                               dims,
	                               D_volumeData, volumePitch);
	if (!ok)
		return false;

	return true;
}


bool ReconAlgo::callFP(float* D_volumeData, unsigned int volumePitch,
                       float* D_projData, unsigned int projPitch,
                       float outputScale)
{
	if (parProjs) {
		assert(!fanProjs);
		return FP(D_volumeData, volumePitch, D_projData, projPitch,
		          dims, parProjs, fOutputScale * outputScale);
	} else {
		assert(fanProjs);
		return FanFP(D_volumeData, volumePitch, D_projData, projPitch,
		             dims, fanProjs, fOutputScale * outputScale);
	}
}

bool ReconAlgo::callBP(float* D_volumeData, unsigned int volumePitch,
                       float* D_projData, unsigned int projPitch,
                       float outputScale)
{
	if (parProjs) {
		assert(!fanProjs);
		return BP(D_volumeData, volumePitch, D_projData, projPitch,
		          dims, parProjs, outputScale);
	} else {
		assert(fanProjs);
		return FanBP(D_volumeData, volumePitch, D_projData, projPitch,
		             dims, fanProjs, outputScale);
	}

}



}
