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

// FIXME: Remove these functions. (Outdated)
__global__ void devMUL_SART(float* pfOut, const float* pfIn, unsigned int pitch, unsigned int width)
{
	unsigned int x = threadIdx.x + 16*blockIdx.x;
	if (x >= width) return;

	pfOut[x] *= pfIn[x];
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

	fRelaxation = 1.0f;
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
	fRelaxation = 1.0f;

	ReconAlgo::reset();
}

bool SART::init()
{
	if (useVolumeMask) {
		allocateVolumeData(D_tmpData, tmpPitch, dims);
		zeroVolumeData(D_tmpData, tmpPitch, dims);
	}

	// NB: Non-standard dimensions
	SDimensions linedims = dims;
	linedims.iProjAngles = 1;
	allocateProjectionData(D_projData, projPitch, linedims);
	zeroProjectionData(D_projData, projPitch, linedims);
	
	allocateProjectionData(D_lineWeight, linePitch, dims);
	zeroProjectionData(D_lineWeight, linePitch, dims);

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
	zeroProjectionData(D_lineWeight, linePitch, dims);
	if (useVolumeMask) {
		callFP(D_maskData, maskPitch, D_lineWeight, linePitch, 1.0f);
	} else {
		// Allocate tmpData temporarily
		allocateVolumeData(D_tmpData, tmpPitch, dims);
		zeroVolumeData(D_tmpData, tmpPitch, dims);


		processVol<opSet>(D_tmpData, 1.0f, tmpPitch, dims);
		callFP(D_tmpData, tmpPitch, D_lineWeight, linePitch, 1.0f);


		cudaFree(D_tmpData);
		D_tmpData = 0;
	}
	processSino<opInvert>(D_lineWeight, linePitch, dims);

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
		// NB: Non-standard dimensions
		SDimensions linedims = dims;
		linedims.iProjAngles = 1;
		duplicateProjectionData(D_projData, D_sinoData + angle*sinoPitch, sinoPitch, linedims);

		// do FP, subtracting projection from sinogram
		if (useVolumeMask) {
				duplicateVolumeData(D_tmpData, D_volumeData, volumePitch, dims);
				processVol<opMul>(D_tmpData, D_maskData, tmpPitch, dims);
				callFP_SART(D_tmpData, tmpPitch, D_projData, projPitch, angle, -1.0f);
		} else {
				callFP_SART(D_volumeData, volumePitch, D_projData, projPitch, angle, -1.0f);
		}

		MUL_SART(D_projData, D_lineWeight + angle*linePitch, projPitch, dims.iProjDets);

		if (useVolumeMask) {
			// BP, mask, and add back
			// TODO: Try putting the masking directly in the BP
			zeroVolumeData(D_tmpData, tmpPitch, dims);
			callBP_SART(D_tmpData, tmpPitch, D_projData, projPitch, angle, fRelaxation);
			processVol<opAddMul>(D_volumeData, D_maskData, D_tmpData, volumePitch, dims);
		} else {
			callBP_SART(D_volumeData, volumePitch, D_projData, projPitch, angle, fRelaxation);
		}

		if (useMinConstraint)
			processVol<opClampMin>(D_volumeData, fMinConstraint, volumePitch, dims);
		if (useMaxConstraint)
			processVol<opClampMax>(D_volumeData, fMaxConstraint, volumePitch, dims);

		iteration++;

	}

	return true;
}

float SART::computeDiffNorm()
{
	unsigned int pPitch;
	float *D_p;
	allocateProjectionData(D_p, pPitch, dims);

	// copy sinogram to D_p
	duplicateProjectionData(D_p, D_sinoData, sinoPitch, dims);

	// do FP, subtracting projection from sinogram
	if (useVolumeMask) {
			duplicateVolumeData(D_tmpData, D_volumeData, volumePitch, dims);
			processVol<opMul>(D_tmpData, D_maskData, tmpPitch, dims);
			callFP(D_tmpData, tmpPitch, D_p, pPitch, -1.0f);
	} else {
			callFP(D_volumeData, volumePitch, D_p, pPitch, -1.0f);
	}


	// compute norm of D_p
	float s = dotProduct2D(D_p, pPitch, dims.iProjDets, dims.iProjAngles);

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
                       unsigned int angle, float outputScale)
{
	if (angles) {
		assert(!fanProjs);
		return BP_SART(D_volumeData, volumePitch, D_projData, projPitch,
		               angle, dims, angles, TOffsets, outputScale);
	} else {
		assert(fanProjs);
		return FanBP_SART(D_volumeData, volumePitch, D_projData, projPitch,
		                  angle, dims, fanProjs, outputScale);
	}

}


}


