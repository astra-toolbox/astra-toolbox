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

#include "sirt.h"
#include "util.h"
#include "arith.h"

#ifdef STANDALONE
#include "testutil.h"
#endif

namespace astraCUDA {

SIRT::SIRT() : ReconAlgo()
{
	D_projData = 0;
	D_tmpData = 0;

	D_lineWeight = 0;
	D_pixelWeight = 0;

	D_minMaskData = 0;
	D_maxMaskData = 0;

	freeMinMaxMasks = false;
}


SIRT::~SIRT()
{
	reset();
}

void SIRT::reset()
{
	cudaFree(D_projData);
	cudaFree(D_tmpData);
	cudaFree(D_lineWeight);
	cudaFree(D_pixelWeight);
	if (freeMinMaxMasks) {
		cudaFree(D_minMaskData);
		cudaFree(D_maxMaskData);
	}

	D_projData = 0;
	D_tmpData = 0;

	D_lineWeight = 0;
	D_pixelWeight = 0;

	freeMinMaxMasks = false;
	D_minMaskData = 0;
	D_maxMaskData = 0;

	useVolumeMask = false;
	useSinogramMask = false;

	ReconAlgo::reset();
}

bool SIRT::init()
{
	allocateVolume(D_pixelWeight, dims.iVolWidth+2, dims.iVolHeight+2, pixelPitch);
	zeroVolume(D_pixelWeight, pixelPitch, dims.iVolWidth+2, dims.iVolHeight+2);

	allocateVolume(D_tmpData, dims.iVolWidth+2, dims.iVolHeight+2, tmpPitch);
	zeroVolume(D_tmpData, tmpPitch, dims.iVolWidth+2, dims.iVolHeight+2);

	allocateVolume(D_projData, dims.iProjDets+2, dims.iProjAngles, projPitch);
	zeroVolume(D_projData, projPitch, dims.iProjDets+2, dims.iProjAngles);
	
	allocateVolume(D_lineWeight, dims.iProjDets+2, dims.iProjAngles, linePitch);
	zeroVolume(D_lineWeight, linePitch, dims.iProjDets+2, dims.iProjAngles);

	// We can't precompute lineWeights and pixelWeights when using a mask
	if (!useVolumeMask && !useSinogramMask)
		precomputeWeights();

	// TODO: check if allocations succeeded
	return true;
}

bool SIRT::precomputeWeights()
{
	zeroVolume(D_lineWeight, linePitch, dims.iProjDets+2, dims.iProjAngles);
	if (useVolumeMask) {
		callFP(D_maskData, maskPitch, D_lineWeight, linePitch, 1.0f);
	} else {
		processVol<opSet, VOL>(D_tmpData, 1.0f, tmpPitch, dims.iVolWidth, dims.iVolHeight);
		callFP(D_tmpData, tmpPitch, D_lineWeight, linePitch, 1.0f);
	}
	processVol<opInvert, SINO>(D_lineWeight, linePitch, dims.iProjDets, dims.iProjAngles);

	if (useSinogramMask) {
		// scale line weights with sinogram mask to zero out masked sinogram pixels
		processVol<opMul, SINO>(D_lineWeight, D_smaskData, linePitch, dims.iProjDets, dims.iProjAngles);
	}


	zeroVolume(D_pixelWeight, pixelPitch, dims.iVolWidth+2, dims.iVolHeight+2);
	if (useSinogramMask) {
		callBP(D_pixelWeight, pixelPitch, D_smaskData, smaskPitch);
	} else {
		processVol<opSet, SINO>(D_projData, 1.0f, projPitch, dims.iProjDets, dims.iProjAngles);
		callBP(D_pixelWeight, pixelPitch, D_projData, projPitch);
	}
	processVol<opInvert, VOL>(D_pixelWeight, pixelPitch, dims.iVolWidth, dims.iVolHeight);

	if (useVolumeMask) {
		// scale pixel weights with mask to zero out masked pixels
		processVol<opMul, VOL>(D_pixelWeight, D_maskData, pixelPitch, dims.iVolWidth, dims.iVolHeight);
	}

	return true;
}

bool SIRT::setMinMaxMasks(float* D_minMaskData_, float* D_maxMaskData_,
	                      unsigned int iPitch)
{
	D_minMaskData = D_minMaskData_;
	D_maxMaskData = D_maxMaskData_;
	minMaskPitch = iPitch;
	maxMaskPitch = iPitch;

	freeMinMaxMasks = false;
	return true;
}

bool SIRT::uploadMinMaxMasks(const float* pfMinMaskData, const float* pfMaxMaskData,
	                         unsigned int iPitch)
{
	freeMinMaxMasks = true;
	bool ok = true;
	if (pfMinMaskData) {
		allocateVolume(D_minMaskData, dims.iVolWidth+2, dims.iVolHeight+2, minMaskPitch);
		ok = copyVolumeToDevice(pfMinMaskData, iPitch,
		                        dims.iVolWidth, dims.iVolHeight,
		                        D_minMaskData, minMaskPitch);
	}
	if (!ok)
		return false;

	if (pfMaxMaskData) {
		allocateVolume(D_maxMaskData, dims.iVolWidth+2, dims.iVolHeight+2, maxMaskPitch);
		ok = copyVolumeToDevice(pfMaxMaskData, iPitch,
		                        dims.iVolWidth, dims.iVolHeight,
		                        D_maxMaskData, maxMaskPitch);
	}
	if (!ok)
		return false;

	return true;
}

bool SIRT::iterate(unsigned int iterations)
{
	shouldAbort = false;

	if (useVolumeMask || useSinogramMask)
		precomputeWeights();

	// iteration
	for (unsigned int iter = 0; iter < iterations && !shouldAbort; ++iter) {

		// copy sinogram to projection data
		cudaMemcpy2D(D_projData, sizeof(float)*projPitch, D_sinoData, sizeof(float)*sinoPitch, sizeof(float)*(dims.iProjDets+2), dims.iProjAngles, cudaMemcpyDeviceToDevice);

		// do FP, subtracting projection from sinogram
		if (useVolumeMask) {
				cudaMemcpy2D(D_tmpData, sizeof(float)*tmpPitch, D_volumeData, sizeof(float)*volumePitch, sizeof(float)*(dims.iVolWidth+2), dims.iVolHeight+2, cudaMemcpyDeviceToDevice);
				processVol<opMul, VOL>(D_tmpData, D_maskData, tmpPitch, dims.iVolWidth, dims.iVolHeight);
				callFP(D_tmpData, tmpPitch, D_projData, projPitch, -1.0f);
		} else {
				callFP(D_volumeData, volumePitch, D_projData, projPitch, -1.0f);
		}

		processVol<opMul, SINO>(D_projData, D_lineWeight, projPitch, dims.iProjDets, dims.iProjAngles);

		zeroVolume(D_tmpData, tmpPitch, dims.iVolWidth+2, dims.iVolHeight+2);

		callBP(D_tmpData, tmpPitch, D_projData, projPitch);

		processVol<opAddMul, VOL>(D_volumeData, D_pixelWeight, D_tmpData, volumePitch, dims.iVolWidth, dims.iVolHeight);

		if (useMinConstraint)
			processVol<opClampMin, VOL>(D_volumeData, fMinConstraint, volumePitch, dims.iVolWidth, dims.iVolHeight);
		if (useMaxConstraint)
			processVol<opClampMax, VOL>(D_volumeData, fMaxConstraint, volumePitch, dims.iVolWidth, dims.iVolHeight);
		if (D_minMaskData)
			processVol<opClampMinMask, VOL>(D_volumeData, D_minMaskData, volumePitch, dims.iVolWidth, dims.iVolHeight);
		if (D_maxMaskData)
			processVol<opClampMaxMask, VOL>(D_volumeData, D_maxMaskData, volumePitch, dims.iVolWidth, dims.iVolHeight);
	}

	return true;
}

float SIRT::computeDiffNorm()
{
	// copy sinogram to projection data
	cudaMemcpy2D(D_projData, sizeof(float)*projPitch, D_sinoData, sizeof(float)*sinoPitch, sizeof(float)*(dims.iProjDets+2), dims.iProjAngles, cudaMemcpyDeviceToDevice);

	// do FP, subtracting projection from sinogram
	if (useVolumeMask) {
			cudaMemcpy2D(D_tmpData, sizeof(float)*tmpPitch, D_volumeData, sizeof(float)*volumePitch, sizeof(float)*(dims.iVolWidth+2), dims.iVolHeight+2, cudaMemcpyDeviceToDevice);
			processVol<opMul, VOL>(D_tmpData, D_maskData, tmpPitch, dims.iVolWidth, dims.iVolHeight);
			callFP(D_tmpData, tmpPitch, D_projData, projPitch, -1.0f);
	} else {
			callFP(D_volumeData, volumePitch, D_projData, projPitch, -1.0f);
	}


	// compute norm of D_projData

	float s = dotProduct2D(D_projData, projPitch, dims.iProjDets, dims.iProjAngles, 1, 0);

	return sqrt(s);
}


bool doSIRT(float* D_volumeData, unsigned int volumePitch,
            float* D_sinoData, unsigned int sinoPitch,
            float* D_maskData, unsigned int maskPitch,
            const SDimensions& dims, const float* angles,
            const float* TOffsets, unsigned int iterations)
{
	SIRT sirt;
	bool ok = true;

	ok &= sirt.setGeometry(dims, angles);
	if (D_maskData)
		ok &= sirt.enableVolumeMask();
	if (TOffsets)
		ok &= sirt.setTOffsets(TOffsets);

	if (!ok)
		return false;

	ok = sirt.init();
	if (!ok)
		return false;

	if (D_maskData)
		ok &= sirt.setVolumeMask(D_maskData, maskPitch);

	ok &= sirt.setBuffers(D_volumeData, volumePitch, D_sinoData, sinoPitch);
	if (!ok)
		return false;

	ok = sirt.iterate(iterations);

	return ok;
}

}

#ifdef STANDALONE

using namespace astraCUDA;

int main()
{
	float* D_volumeData;
	float* D_sinoData;

	SDimensions dims;
	dims.iVolWidth = 1024;
	dims.iVolHeight = 1024;
	dims.iProjAngles = 512;
	dims.iProjDets = 1536;
	dims.fDetScale = 1.0f;
	dims.iRaysPerDet = 1;
	unsigned int volumePitch, sinoPitch;

	allocateVolume(D_volumeData, dims.iVolWidth+2, dims.iVolHeight+2, volumePitch);
	zeroVolume(D_volumeData, volumePitch, dims.iVolWidth+2, dims.iVolHeight+2);
	printf("pitch: %u\n", volumePitch);

	allocateVolume(D_sinoData, dims.iProjDets+2, dims.iProjAngles, sinoPitch);
	zeroVolume(D_sinoData, sinoPitch, dims.iProjDets+2, dims.iProjAngles);
	printf("pitch: %u\n", sinoPitch);
	
	unsigned int y, x;
	float* sino = loadImage("sino.png", y, x);

	float* img = new float[dims.iVolWidth*dims.iVolHeight];

	copySinogramToDevice(sino, dims.iProjDets, dims.iProjDets, dims.iProjAngles, D_sinoData, sinoPitch);

	float* angle = new float[dims.iProjAngles];

	for (unsigned int i = 0; i < dims.iProjAngles; ++i)
		angle[i] = i*(M_PI/dims.iProjAngles);

	SIRT sirt;

	sirt.setGeometry(dims, angle);
	sirt.init();

	sirt.setBuffers(D_volumeData, volumePitch, D_sinoData, sinoPitch);

	sirt.iterate(25);


	delete[] angle;

	copyVolumeFromDevice(img, dims.iVolWidth, dims.iVolWidth, dims.iVolHeight, D_volumeData, volumePitch);

	saveImage("vol.png",dims.iVolHeight,dims.iVolWidth,img);

	return 0;
}
#endif

