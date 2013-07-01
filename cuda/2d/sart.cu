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

#include "sart.h"
#include "util.h"
#include "arith.h"
#include "fan_fp.h"
#include "fan_bp.h"
#include "par_fp.h"
#include "par_bp.h"

namespace astraCUDA {


__global__ void devMUL_SART(float* pfOut, const float* pfIn, unsigned int pitch, unsigned int width)
{
	unsigned int x = threadIdx.x + 16*blockIdx.x;
	if (x >= width) return;

	// Copy result down and left one pixel.
	pfOut[x + pitch] = pfOut[x + 1] * pfIn[x + 1];
}

void MUL_SART(float* pfOut, const float* pfIn, unsigned int pitch, unsigned int width)
{
	dim3 blockSize(16,16);
	dim3 gridSize((width+15)/16, 1);

	devMUL_SART<<<gridSize, blockSize>>>(pfOut, pfIn, pitch, width);

	cudaTextForceKernelsCompletion();
}



SART::SART() : ReconAlgo()
{
	D_projData = 0;
	D_tmpData = 0;

	D_lineWeight = 0;

	projectionOrder = 0;
	projectionCount = 0;
	iteration = 0;
	customOrder = false;
}


SART::~SART()
{
	reset();
}

void SART::reset()
{
	cudaFree(D_projData);
	cudaFree(D_tmpData);
	cudaFree(D_lineWeight);

	D_projData = 0;
	D_tmpData = 0;

	D_lineWeight = 0;

	useVolumeMask = false;
	useSinogramMask = false;

	if (projectionOrder != NULL) delete[] projectionOrder;
	projectionOrder = 0;
	projectionCount = 0;
	iteration = 0;
	customOrder = false;

	ReconAlgo::reset();
}

bool SART::init()
{
	if (useVolumeMask) {
		allocateVolume(D_tmpData, dims.iVolWidth+2, dims.iVolHeight+2, tmpPitch);
		zeroVolume(D_tmpData, tmpPitch, dims.iVolWidth+2, dims.iVolHeight+2);
	}

	// HACK: D_projData consists of two lines. The first is used padded,
	// the second unpadded. This is to satisfy the alignment requirements
	// of resp. FP and BP_SART.
	allocateVolume(D_projData, dims.iProjDets+2, 2, projPitch);
	zeroVolume(D_projData, projPitch, dims.iProjDets+2, 1);
	
	allocateVolume(D_lineWeight, dims.iProjDets+2, dims.iProjAngles, linePitch);
	zeroVolume(D_lineWeight, linePitch, dims.iProjDets+2, dims.iProjAngles);

	// We can't precompute lineWeights when using a mask
	if (!useVolumeMask)
		precomputeWeights();

	// TODO: check if allocations succeeded
	return true;
}

bool SART::setProjectionOrder(int* _projectionOrder, int _projectionCount)
{
	customOrder = true;
	projectionCount = _projectionCount;
	projectionOrder = new int[projectionCount];
	for (int i = 0; i < projectionCount; i++) {
		projectionOrder[i] = _projectionOrder[i];
	}

	return true;
}


bool SART::precomputeWeights()
{
	zeroVolume(D_lineWeight, linePitch, dims.iProjDets+2, dims.iProjAngles);
	if (useVolumeMask) {
		callFP(D_maskData, maskPitch, D_lineWeight, linePitch, 1.0f);
	} else {
		// Allocate tmpData temporarily
		allocateVolume(D_tmpData, dims.iVolWidth+2, dims.iVolHeight+2, tmpPitch);
		zeroVolume(D_tmpData, tmpPitch, dims.iVolWidth+2, dims.iVolHeight+2);


		processVol<opSet, VOL>(D_tmpData, 1.0f, tmpPitch, dims.iVolWidth, dims.iVolHeight);
		callFP(D_tmpData, tmpPitch, D_lineWeight, linePitch, 1.0f);


		cudaFree(D_tmpData);
		D_tmpData = 0;
	}
	processVol<opInvert, SINO>(D_lineWeight, linePitch, dims.iProjDets, dims.iProjAngles);

	return true;
}

bool SART::iterate(unsigned int iterations)
{
	shouldAbort = false;

	if (useVolumeMask)
		precomputeWeights();

	// iteration
	for (unsigned int iter = 0; iter < iterations && !shouldAbort; ++iter) {

		int angle;
		if (customOrder) {
			angle = projectionOrder[iteration % projectionCount];
		} else {
			angle = iteration % dims.iProjAngles;  
		}

		// copy one line of sinogram to projection data
		cudaMemcpy2D(D_projData, sizeof(float)*projPitch, D_sinoData + angle*sinoPitch, sizeof(float)*sinoPitch, sizeof(float)*(dims.iProjDets+2), 1, cudaMemcpyDeviceToDevice);

		// do FP, subtracting projection from sinogram
		if (useVolumeMask) {
				cudaMemcpy2D(D_tmpData, sizeof(float)*tmpPitch, D_volumeData, sizeof(float)*volumePitch, sizeof(float)*(dims.iVolWidth+2), dims.iVolHeight+2, cudaMemcpyDeviceToDevice);
				processVol<opMul, VOL>(D_tmpData, D_maskData, tmpPitch, dims.iVolWidth, dims.iVolHeight);
				callFP_SART(D_tmpData, tmpPitch, D_projData, projPitch, angle, -1.0f);
		} else {
				callFP_SART(D_volumeData, volumePitch, D_projData, projPitch, angle, -1.0f);
		}

		MUL_SART(D_projData, D_lineWeight + angle*linePitch, projPitch, dims.iProjDets);

		if (useVolumeMask) {
			// BP, mask, and add back
			// TODO: Try putting the masking directly in the BP
			zeroVolume(D_tmpData, tmpPitch, dims.iVolWidth+2, dims.iVolHeight+2);
			callBP_SART(D_tmpData, tmpPitch, D_projData, projPitch, angle);
			processVol<opAddMul, VOL>(D_volumeData, D_maskData, D_tmpData, volumePitch, dims.iVolWidth, dims.iVolHeight);
		} else {
			callBP_SART(D_volumeData, volumePitch, D_projData, projPitch, angle);
		}

		if (useMinConstraint)
			processVol<opClampMin, VOL>(D_volumeData, fMinConstraint, volumePitch, dims.iVolWidth, dims.iVolHeight);
		if (useMaxConstraint)
			processVol<opClampMax, VOL>(D_volumeData, fMaxConstraint, volumePitch, dims.iVolWidth, dims.iVolHeight);

		iteration++;

	}

	return true;
}

float SART::computeDiffNorm()
{
	unsigned int pPitch;
	float *D_p;
	allocateVolume(D_p, dims.iProjDets+2, dims.iProjAngles, pPitch);
	zeroVolume(D_p, pPitch, dims.iProjDets+2, dims.iProjAngles);

	// copy sinogram to D_p
	cudaMemcpy2D(D_p, sizeof(float)*pPitch, D_sinoData, sizeof(float)*sinoPitch, sizeof(float)*(dims.iProjDets+2), dims.iProjAngles, cudaMemcpyDeviceToDevice);

	// do FP, subtracting projection from sinogram
	if (useVolumeMask) {
			cudaMemcpy2D(D_tmpData, sizeof(float)*tmpPitch, D_volumeData, sizeof(float)*volumePitch, sizeof(float)*(dims.iVolWidth+2), dims.iVolHeight+2, cudaMemcpyDeviceToDevice);
			processVol<opMul, VOL>(D_tmpData, D_maskData, tmpPitch, dims.iVolWidth, dims.iVolHeight);
			callFP(D_tmpData, tmpPitch, D_projData, projPitch, -1.0f);
	} else {
			callFP(D_volumeData, volumePitch, D_projData, projPitch, -1.0f);
	}


	// compute norm of D_p
	float s = dotProduct2D(D_p, pPitch, dims.iProjDets, dims.iProjAngles, 1, 0);

	cudaFree(D_p);

	return sqrt(s);
}

bool SART::callFP_SART(float* D_volumeData, unsigned int volumePitch,
                       float* D_projData, unsigned int projPitch,
                       unsigned int angle, float outputScale)
{
	SDimensions d = dims;
	d.iProjAngles = 1;
	if (angles) {
		assert(!fanProjs);
		return FP(D_volumeData, volumePitch, D_projData, projPitch,
		          d, &angles[angle], TOffsets, outputScale);
	} else {
		assert(fanProjs);
		return FanFP(D_volumeData, volumePitch, D_projData, projPitch,
		             d, &fanProjs[angle], outputScale);
	}
}

bool SART::callBP_SART(float* D_volumeData, unsigned int volumePitch,
                       float* D_projData, unsigned int projPitch,
                       unsigned int angle)
{
	if (angles) {
		assert(!fanProjs);
		return BP_SART(D_volumeData, volumePitch, D_projData + projPitch, projPitch,
		               angle, dims, angles, TOffsets);
	} else {
		assert(fanProjs);
		return FanBP_SART(D_volumeData, volumePitch, D_projData + projPitch, projPitch,
		                  angle, dims, fanProjs);
	}

}


}


