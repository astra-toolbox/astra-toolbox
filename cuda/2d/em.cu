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

#include "em.h"
#include "util.h"
#include "arith.h"

#ifdef STANDALONE
#include "testutil.h"
#endif

namespace astraCUDA {


// TODO: ensure non-negativity somewhere??


EM::EM()
{
	D_projData = 0;
	D_tmpData = 0;
	D_pixelWeight = 0;

}


EM::~EM()
{
	reset();
}

void EM::reset()
{
	cudaFree(D_projData);
	cudaFree(D_tmpData);
	cudaFree(D_pixelWeight);

	D_projData = 0;
	D_tmpData = 0;
	D_pixelWeight = 0;

	ReconAlgo::reset();
}


bool EM::init()
{
	allocateVolumeData(D_pixelWeight, pixelPitch, dims);
	zeroVolumeData(D_pixelWeight, pixelPitch, dims);

	allocateVolumeData(D_tmpData, tmpPitch, dims);
	zeroVolumeData(D_tmpData, tmpPitch, dims);

	allocateProjectionData(D_projData, projPitch, dims);
	zeroProjectionData(D_projData, projPitch, dims);

	// We can't precompute pixelWeights when using a volume mask
#if 0 
	if (!useVolumeMask)
#endif
		precomputeWeights();

	// TODO: check if allocations succeeded
	return true;
}

bool EM::precomputeWeights()
{
	zeroVolumeData(D_pixelWeight, pixelPitch, dims);
#if 0
	if (useSinogramMask) {
		callBP(D_pixelWeight, pixelPitch, D_smaskData, smaskPitch);
	} else
#endif
	{
		processVol<opSet>(D_projData, 1.0f, projPitch, dims.iProjDets, dims.iProjAngles);
		callBP(D_pixelWeight, pixelPitch, D_projData, projPitch);
	}
	processVol<opInvert>(D_pixelWeight, pixelPitch, dims.iVolWidth, dims.iVolHeight);

#if 0
	if (useVolumeMask) {
		// scale pixel weights with mask to zero out masked pixels
		processVol<opMul>(D_pixelWeight, D_maskData, pixelPitch, dims.iVolWidth, dims.iVolHeight);
	}
#endif

	return true;
}

bool EM::iterate(unsigned int iterations)
{
	shouldAbort = false;

#if 0
	if (useVolumeMask)
		precomputeWeights();
#endif

	// iteration
	for (unsigned int iter = 0; iter < iterations && !shouldAbort; ++iter) {

		// Do FP of volumeData 
		zeroProjectionData(D_projData, projPitch, dims);
		callFP(D_volumeData, volumePitch, D_projData, projPitch, 1.0f);

		// Divide sinogram by FP (into projData)
		processVol<opDividedBy>(D_projData, D_sinoData, projPitch, dims.iProjDets, dims.iProjAngles);

		// Do BP of projData into tmpData
		zeroVolumeData(D_tmpData, tmpPitch, dims);
		callBP(D_tmpData, tmpPitch, D_projData, projPitch);

		// Multiply volumeData with tmpData divided by pixel weights
		processVol<opMul2>(D_volumeData, D_tmpData, D_pixelWeight, pixelPitch, dims.iVolWidth, dims.iVolHeight);

	}

	return true;
}

float EM::computeDiffNorm()
{
	// copy sinogram to projection data
	cudaMemcpy2D(D_projData, sizeof(float)*projPitch, D_sinoData, sizeof(float)*sinoPitch, sizeof(float)*(dims.iProjDets), dims.iProjAngles, cudaMemcpyDeviceToDevice);

	// do FP, subtracting projection from sinogram
	if (useVolumeMask) {
			cudaMemcpy2D(D_tmpData, sizeof(float)*tmpPitch, D_volumeData, sizeof(float)*volumePitch, sizeof(float)*(dims.iVolWidth), dims.iVolHeight, cudaMemcpyDeviceToDevice);
			processVol<opMul>(D_tmpData, D_maskData, tmpPitch, dims.iVolWidth, dims.iVolHeight);
			callFP(D_tmpData, tmpPitch, D_projData, projPitch, -1.0f);
	} else {
			callFP(D_volumeData, volumePitch, D_projData, projPitch, -1.0f);
	}


	// compute norm of D_projData

	float s = dotProduct2D(D_projData, projPitch, dims.iProjDets, dims.iProjAngles);

	return sqrt(s);
}


bool doEM(float* D_volumeData, unsigned int volumePitch,
          float* D_sinoData, unsigned int sinoPitch,
          const SDimensions& dims, const float* angles,
          const float* TOffsets, unsigned int iterations)
{
	EM em;
	bool ok = true;

	ok &= em.setGeometry(dims, angles);
	if (TOffsets)
		ok &= em.setTOffsets(TOffsets);

	if (!ok)
		return false;

	ok = em.init();
	if (!ok)
		return false;

	ok &= em.setBuffers(D_volumeData, volumePitch, D_sinoData, sinoPitch);
	if (!ok)
		return false;

	ok = em.iterate(iterations);

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

	allocateVolume(D_volumeData, dims.iVolWidth, dims.iVolHeight, volumePitch);
	zeroVolume(D_volumeData, volumePitch, dims.iVolWidth, dims.iVolHeight);
	printf("pitch: %u\n", volumePitch);

	allocateVolume(D_sinoData, dims.iProjDets, dims.iProjAngles, sinoPitch);
	zeroVolume(D_sinoData, sinoPitch, dims.iProjDets, dims.iProjAngles);
	printf("pitch: %u\n", sinoPitch);
	
	unsigned int y, x;
	float* sino = loadImage("sino.png", y, x);

	float* img = new float[dims.iVolWidth*dims.iVolHeight];

	copySinogramToDevice(sino, dims.iProjDets, dims.iProjDets, dims.iProjAngles, D_sinoData, sinoPitch);

	float* angle = new float[dims.iProjAngles];

	for (unsigned int i = 0; i < dims.iProjAngles; ++i)
		angle[i] = i*(M_PI/dims.iProjAngles);

	EM em;

	em.setGeometry(dims, angle);
	em.init();

	// TODO: Initialize D_volumeData with an unfiltered backprojection

	em.setBuffers(D_volumeData, volumePitch, D_sinoData, sinoPitch);

	em.iterate(25);


	delete[] angle;

	copyVolumeFromDevice(img, dims.iVolWidth, dims.iVolWidth, dims.iVolHeight, D_volumeData, volumePitch);

	saveImage("vol.png",dims.iVolHeight,dims.iVolWidth,img);

	return 0;
}

#endif
