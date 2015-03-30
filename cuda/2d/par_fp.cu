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
#include <iostream>
#include <list>

#include "util.h"
#include "arith.h"

#ifdef STANDALONE
#include "testutil.h"
#endif

#define PIXELTRACE


typedef texture<float, 2, cudaReadModeElementType> texture2D;

static texture2D gT_volumeTexture;


namespace astraCUDA {

static const unsigned g_MaxAngles = 2560;
__constant__ float gC_angle[g_MaxAngles];
__constant__ float gC_angle_offset[g_MaxAngles];


// optimization parameters
static const unsigned int g_anglesPerBlock = 16;
static const unsigned int g_detBlockSize = 32;
static const unsigned int g_blockSlices = 64;

// fixed point scaling factor
#define fPREC_FACTOR 16.0f
#define iPREC_FACTOR 16


// if necessary, a buffer of zeroes of size g_MaxAngles
static float* g_pfZeroes = 0;


static bool bindVolumeDataTexture(float* data, cudaArray*& dataArray, unsigned int pitch, unsigned int width, unsigned int height)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	dataArray = 0;
	cudaMallocArray(&dataArray, &channelDesc, width, height);
	cudaMemcpy2DToArray(dataArray, 0, 0, data, pitch*sizeof(float), width*sizeof(float), height, cudaMemcpyDeviceToDevice);

	gT_volumeTexture.addressMode[0] = cudaAddressModeBorder;
	gT_volumeTexture.addressMode[1] = cudaAddressModeBorder;
	gT_volumeTexture.filterMode = cudaFilterModeLinear;
	gT_volumeTexture.normalized = false;

	// TODO: For very small sizes (roughly <=512x128) with few angles (<=180)
	// not using an array is more efficient.
//	cudaBindTexture2D(0, gT_volumeTexture, (const void*)data, channelDesc, width, height, sizeof(float)*pitch);
	cudaBindTextureToArray(gT_volumeTexture, dataArray, channelDesc);

	// TODO: error value?

	return true;
}

// projection for angles that are roughly horizontal
// theta between 45 and 135 degrees
__global__ void FPhorizontal(float* D_projData, unsigned int projPitch, unsigned int startSlice, unsigned int startAngle, unsigned int endAngle, int regionOffset, const SDimensions dims, float outputScale)
{
	const int relDet = threadIdx.x;
	const int relAngle = threadIdx.y;

	int angle = startAngle + blockIdx.x * g_anglesPerBlock + relAngle;

	if (angle >= endAngle)
		return;

	const float theta = gC_angle[angle];
	const float cos_theta = __cosf(theta);
	const float sin_theta = __sinf(theta);

	// compute start detector for this block/angle:
	// (The same for all threadIdx.x)
	// -------------------------------------

	const int midSlice = startSlice + g_blockSlices / 2;

	// ASSUMPTION: fDetScale >= 1.0f
	// problem: detector regions get skipped because slice blocks aren't large
	// enough
	const unsigned int g_blockSliceSize = g_detBlockSize;

	// project (midSlice,midRegion) on this thread's detector

	const float fBase = 0.5f*dims.iProjDets - 0.5f +
		(
		    (midSlice - 0.5f*dims.iVolWidth + 0.5f) * cos_theta
		  - (g_blockSliceSize/2 - 0.5f*dims.iVolHeight + 0.5f) * sin_theta
		  + gC_angle_offset[angle]
		) / dims.fDetScale;
	int iBase = (int)floorf(fBase * fPREC_FACTOR);
	int iInc = (int)floorf(g_blockSliceSize * sin_theta / dims.fDetScale * -fPREC_FACTOR);

	// ASSUMPTION: 16 > regionOffset / fDetScale
	const int detRegion = (iBase + (blockIdx.y - regionOffset)*iInc + 16*iPREC_FACTOR*g_detBlockSize) / (iPREC_FACTOR * g_detBlockSize) - 16;
	const int detPrevRegion = (iBase + (blockIdx.y - regionOffset - 1)*iInc + 16*iPREC_FACTOR*g_detBlockSize) / (iPREC_FACTOR * g_detBlockSize) - 16;

	if (blockIdx.y > 0 && detRegion == detPrevRegion)
		return;

	const int detector = detRegion * g_detBlockSize + relDet;

	// Now project the part of the ray to angle,detector through
	// slices startSlice to startSlice+g_blockSlices-1

	if (detector < 0 || detector >= dims.iProjDets)
		return;

	const float fDetStep = -dims.fDetScale / sin_theta;
	float fSliceStep = cos_theta / sin_theta;
	float fDistCorr;
	if (sin_theta > 0.0f)
		fDistCorr = -fDetStep;
	else
		fDistCorr = fDetStep;
	fDistCorr *= outputScale;

	float fVal = 0.0f;
	// project detector on slice
	float fP = (detector - 0.5f*dims.iProjDets + 0.5f - gC_angle_offset[angle]) * fDetStep + (startSlice - 0.5f*dims.iVolWidth + 0.5f) * fSliceStep + 0.5f*dims.iVolHeight - 0.5f + 0.5f;
	float fS = startSlice + 0.5f;
	int endSlice = startSlice + g_blockSlices;
	if (endSlice > dims.iVolWidth)
		endSlice = dims.iVolWidth;

	if (dims.iRaysPerDet > 1) {

		fP += (-0.5f*dims.iRaysPerDet + 0.5f)/dims.iRaysPerDet * fDetStep;
		const float fSubDetStep = fDetStep / dims.iRaysPerDet;
		fDistCorr /= dims.iRaysPerDet;

		fSliceStep -= dims.iRaysPerDet * fSubDetStep;

		for (int slice = startSlice; slice < endSlice; ++slice)
		{
			for (int iSubT = 0; iSubT < dims.iRaysPerDet; ++iSubT) {
				fVal += tex2D(gT_volumeTexture, fS, fP);
				fP += fSubDetStep;
			}
			fP += fSliceStep;
			fS += 1.0f;
		}

	} else {

		for (int slice = startSlice; slice < endSlice; ++slice)
		{
			fVal += tex2D(gT_volumeTexture, fS, fP);
			fP += fSliceStep;
			fS += 1.0f;
		}


	}

	D_projData[angle*projPitch+detector] += fVal * fDistCorr;
}

// projection for angles that are roughly vertical
// theta between 0 and 45, or 135 and 180 degrees
__global__ void FPvertical(float* D_projData, unsigned int projPitch, unsigned int startSlice, unsigned int startAngle, unsigned int endAngle, int regionOffset, const SDimensions dims, float outputScale)
{
	const int relDet = threadIdx.x;
	const int relAngle = threadIdx.y;

	int angle = startAngle + blockIdx.x * g_anglesPerBlock + relAngle;

	if (angle >= endAngle)
		return;

	const float theta = gC_angle[angle];
	const float cos_theta = __cosf(theta);
	const float sin_theta = __sinf(theta);

	// compute start detector for this block/angle:
	// (The same for all threadIdx.x)
	// -------------------------------------

	const int midSlice = startSlice + g_blockSlices / 2;

	// project (midSlice,midRegion) on this thread's detector

	// ASSUMPTION: fDetScale >= 1.0f
	// problem: detector regions get skipped because slice blocks aren't large
	// enough
	const unsigned int g_blockSliceSize = g_detBlockSize;

	const float fBase = 0.5f*dims.iProjDets - 0.5f +
		(
		    (g_blockSliceSize/2 - 0.5f*dims.iVolWidth + 0.5f) * cos_theta
		  - (midSlice - 0.5f*dims.iVolHeight + 0.5f) * sin_theta
		  + gC_angle_offset[angle]
		) / dims.fDetScale;
	int iBase = (int)floorf(fBase * fPREC_FACTOR);
	int iInc = (int)floorf(g_blockSliceSize * cos_theta / dims.fDetScale * fPREC_FACTOR);

	// ASSUMPTION: 16 > regionOffset / fDetScale
	const int detRegion = (iBase + (blockIdx.y - regionOffset)*iInc + 16*iPREC_FACTOR*g_detBlockSize) / (iPREC_FACTOR * g_detBlockSize) - 16;
	const int detPrevRegion = (iBase + (blockIdx.y - regionOffset-1)*iInc + 16*iPREC_FACTOR*g_detBlockSize) / (iPREC_FACTOR * g_detBlockSize) - 16;

	if (blockIdx.y > 0 && detRegion == detPrevRegion)
		return;

	const int detector = detRegion * g_detBlockSize + relDet;

	// Now project the part of the ray to angle,detector through
	// slices startSlice to startSlice+g_blockSlices-1

	if (detector < 0 || detector >= dims.iProjDets)
		return;

	const float fDetStep = dims.fDetScale / cos_theta;
	float fSliceStep = sin_theta / cos_theta;
	float fDistCorr;
	if (cos_theta < 0.0f)
		fDistCorr = -fDetStep;
	else
		fDistCorr = fDetStep;
	fDistCorr *= outputScale;

	float fVal = 0.0f;
	float fP = (detector - 0.5f*dims.iProjDets + 0.5f - gC_angle_offset[angle]) * fDetStep + (startSlice - 0.5f*dims.iVolHeight + 0.5f) * fSliceStep + 0.5f*dims.iVolWidth - 0.5f + 0.5f;
	float fS = startSlice+0.5f;
	int endSlice = startSlice + g_blockSlices;
	if (endSlice > dims.iVolHeight)
		endSlice = dims.iVolHeight;

	if (dims.iRaysPerDet > 1) {

		fP += (-0.5f*dims.iRaysPerDet + 0.5f)/dims.iRaysPerDet * fDetStep;
		const float fSubDetStep = fDetStep / dims.iRaysPerDet;
		fDistCorr /= dims.iRaysPerDet;

		fSliceStep -= dims.iRaysPerDet * fSubDetStep;

		for (int slice = startSlice; slice < endSlice; ++slice)
		{
			for (int iSubT = 0; iSubT < dims.iRaysPerDet; ++iSubT) {
				fVal += tex2D(gT_volumeTexture, fP, fS);
				fP += fSubDetStep;
			}
			fP += fSliceStep;
			fS += 1.0f;
		}

	} else {

		for (int slice = startSlice; slice < endSlice; ++slice)
		{
			fVal += tex2D(gT_volumeTexture, fP, fS);
			fP += fSliceStep;
			fS += 1.0f;
		}

	}

	D_projData[angle*projPitch+detector] += fVal * fDistCorr;
}

// projection for angles that are roughly horizontal
// (detector roughly vertical)
__global__ void FPhorizontal_simple(float* D_projData, unsigned int projPitch, unsigned int startSlice, unsigned int startAngle, unsigned int endAngle, const SDimensions dims, float outputScale)
{
	const int relDet = threadIdx.x;
	const int relAngle = threadIdx.y;

	int angle = startAngle + blockIdx.x * g_anglesPerBlock + relAngle;

	if (angle >= endAngle)
		return;

	const float theta = gC_angle[angle];
	const float cos_theta = __cosf(theta);
	const float sin_theta = __sinf(theta);

	// compute start detector for this block/angle:
	const int detRegion = blockIdx.y;

	const int detector = detRegion * g_detBlockSize + relDet;

	// Now project the part of the ray to angle,detector through
	// slices startSlice to startSlice+g_blockSlices-1

	if (detector < 0 || detector >= dims.iProjDets)
		return;

	const float fDetStep = -dims.fDetScale / sin_theta;
	float fSliceStep = cos_theta / sin_theta;
	float fDistCorr;
	if (sin_theta > 0.0f)
		fDistCorr = -fDetStep;
	else
		fDistCorr = fDetStep;
	fDistCorr *= outputScale;

	float fVal = 0.0f;
	// project detector on slice
	float fP = (detector - 0.5f*dims.iProjDets + 0.5f - gC_angle_offset[angle]) * fDetStep + (startSlice - 0.5f*dims.iVolWidth + 0.5f) * fSliceStep + 0.5f*dims.iVolHeight - 0.5f + 0.5f;
	float fS = startSlice + 0.5f;
	int endSlice = startSlice + g_blockSlices;
	if (endSlice > dims.iVolWidth)
		endSlice = dims.iVolWidth;

	if (dims.iRaysPerDet > 1) {

		fP += (-0.5f*dims.iRaysPerDet + 0.5f)/dims.iRaysPerDet * fDetStep;
		const float fSubDetStep = fDetStep / dims.iRaysPerDet;
		fDistCorr /= dims.iRaysPerDet;

		fSliceStep -= dims.iRaysPerDet * fSubDetStep;

		for (int slice = startSlice; slice < endSlice; ++slice)
		{
			for (int iSubT = 0; iSubT < dims.iRaysPerDet; ++iSubT) {
				fVal += tex2D(gT_volumeTexture, fS, fP);
				fP += fSubDetStep;
			}
			fP += fSliceStep;
			fS += 1.0f;
		}

	} else {

		for (int slice = startSlice; slice < endSlice; ++slice)
		{
			fVal += tex2D(gT_volumeTexture, fS, fP);
			fP += fSliceStep;
			fS += 1.0f;
		}


	}

	D_projData[angle*projPitch+detector] += fVal * fDistCorr;
}


// projection for angles that are roughly vertical
// (detector roughly horizontal)
__global__ void FPvertical_simple(float* D_projData, unsigned int projPitch, unsigned int startSlice, unsigned int startAngle, unsigned int endAngle, const SDimensions dims, float outputScale)
{
	const int relDet = threadIdx.x;
	const int relAngle = threadIdx.y;

	int angle = startAngle + blockIdx.x * g_anglesPerBlock + relAngle;

	if (angle >= endAngle)
		return;

	const float theta = gC_angle[angle];
	const float cos_theta = __cosf(theta);
	const float sin_theta = __sinf(theta);

	// compute start detector for this block/angle:
	const int detRegion = blockIdx.y;

	const int detector = detRegion * g_detBlockSize + relDet;

	// Now project the part of the ray to angle,detector through
	// slices startSlice to startSlice+g_blockSlices-1

	if (detector < 0 || detector >= dims.iProjDets)
		return;

	const float fDetStep = dims.fDetScale / cos_theta;
	float fSliceStep = sin_theta / cos_theta;
	float fDistCorr;
	if (cos_theta < 0.0f)
		fDistCorr = -fDetStep;
	else
		fDistCorr = fDetStep;
	fDistCorr *= outputScale;

	float fVal = 0.0f;
	float fP = (detector - 0.5f*dims.iProjDets + 0.5f - gC_angle_offset[angle]) * fDetStep + (startSlice - 0.5f*dims.iVolHeight + 0.5f) * fSliceStep + 0.5f*dims.iVolWidth - 0.5f + 0.5f;
	float fS = startSlice+0.5f;
	int endSlice = startSlice + g_blockSlices;
	if (endSlice > dims.iVolHeight)
		endSlice = dims.iVolHeight;

	if (dims.iRaysPerDet > 1) {

		fP += (-0.5f*dims.iRaysPerDet + 0.5f)/dims.iRaysPerDet * fDetStep;
		const float fSubDetStep = fDetStep / dims.iRaysPerDet;
		fDistCorr /= dims.iRaysPerDet;

		fSliceStep -= dims.iRaysPerDet * fSubDetStep;

		for (int slice = startSlice; slice < endSlice; ++slice)
		{
			for (int iSubT = 0; iSubT < dims.iRaysPerDet; ++iSubT) {
				fVal += tex2D(gT_volumeTexture, fP, fS);
				fP += fSubDetStep;
			}
			fP += fSliceStep;
			fS += 1.0f;
		}

	} else {

		for (int slice = startSlice; slice < endSlice; ++slice)
		{
			fVal += tex2D(gT_volumeTexture, fP, fS);
			fP += fSliceStep;
			fS += 1.0f;
		}

	}

	D_projData[angle*projPitch+detector] += fVal * fDistCorr;
}



bool FP_simple_internal(float* D_volumeData, unsigned int volumePitch,
               float* D_projData, unsigned int projPitch,
               const SDimensions& dims, const float* angles,
               const float* TOffsets, float outputScale)
{
	// TODO: load angles into constant memory in smaller blocks
	assert(dims.iProjAngles <= g_MaxAngles);

	cudaArray* D_dataArray;
	bindVolumeDataTexture(D_volumeData, D_dataArray, volumePitch, dims.iVolWidth, dims.iVolHeight);

	cudaMemcpyToSymbol(gC_angle, angles, dims.iProjAngles*sizeof(float), 0, cudaMemcpyHostToDevice);

	if (TOffsets) {
		cudaMemcpyToSymbol(gC_angle_offset, TOffsets, dims.iProjAngles*sizeof(float), 0, cudaMemcpyHostToDevice);
	} else {
		if (!g_pfZeroes) {
			g_pfZeroes = new float[g_MaxAngles];
			memset(g_pfZeroes, 0, g_MaxAngles * sizeof(float));
		}
		cudaMemcpyToSymbol(gC_angle_offset, g_pfZeroes, dims.iProjAngles*sizeof(float), 0, cudaMemcpyHostToDevice);
	}

	dim3 dimBlock(g_detBlockSize, g_anglesPerBlock); // detector block size, angles

	std::list<cudaStream_t> streams;


	// Run over all angles, grouping them into groups of the same
	// orientation (roughly horizontal vs. roughly vertical).
	// Start a stream of grids for each such group.

	// TODO: Check if it's worth it to store this info instead
	// of recomputing it every FP.

	unsigned int blockStart = 0;
	unsigned int blockEnd = 0;
	bool blockVertical = false;
	for (unsigned int a = 0; a <= dims.iProjAngles; ++a) {
		bool vertical = false;
		// TODO: Having <= instead of < below causes a 5% speedup.
		// Maybe we should detect corner cases and put them in the optimal
		// group of angles.
		if (a != dims.iProjAngles)
			vertical = (fabsf(sinf(angles[a])) <= fabsf(cosf(angles[a])));
		if (a == dims.iProjAngles || vertical != blockVertical) {
			// block done

			blockEnd = a;
			if (blockStart != blockEnd) {
				dim3 dimGrid((blockEnd-blockStart+g_anglesPerBlock-1)/g_anglesPerBlock,
				             (dims.iProjDets+g_detBlockSize-1)/g_detBlockSize); // angle blocks, detector blocks

				// TODO: check if we can't immediately
				//       destroy the stream after use
				cudaStream_t stream;
				cudaStreamCreate(&stream);
				streams.push_back(stream);
				//printf("angle block: %d to %d, %d\n", blockStart, blockEnd, blockVertical);
				if (!blockVertical)
					for (unsigned int i = 0; i < dims.iVolWidth; i += g_blockSlices)
						FPhorizontal_simple<<<dimGrid, dimBlock, 0, stream>>>(D_projData, projPitch, i, blockStart, blockEnd, dims, outputScale);
				else
					for (unsigned int i = 0; i < dims.iVolHeight; i += g_blockSlices)
						FPvertical_simple<<<dimGrid, dimBlock, 0, stream>>>(D_projData, projPitch, i, blockStart, blockEnd, dims, outputScale);
			}
			blockVertical = vertical;
			blockStart = a;
		}
	}

	for (std::list<cudaStream_t>::iterator iter = streams.begin(); iter != streams.end(); ++iter)
		cudaStreamDestroy(*iter);

	streams.clear();

	cudaThreadSynchronize();

	cudaTextForceKernelsCompletion();

	cudaFreeArray(D_dataArray);
		

	return true;
}

bool FP_simple(float* D_volumeData, unsigned int volumePitch,
               float* D_projData, unsigned int projPitch,
               const SDimensions& dims, const float* angles,
               const float* TOffsets, float outputScale)
{
	for (unsigned int iAngle = 0; iAngle < dims.iProjAngles; iAngle += g_MaxAngles) {
		SDimensions subdims = dims;
		unsigned int iEndAngle = iAngle + g_MaxAngles;
		if (iEndAngle >= dims.iProjAngles)
			iEndAngle = dims.iProjAngles;
		subdims.iProjAngles = iEndAngle - iAngle;

		bool ret;
		ret = FP_simple_internal(D_volumeData, volumePitch,
		                         D_projData + iAngle * projPitch, projPitch,
		                         subdims, angles + iAngle,
		                         TOffsets ? TOffsets + iAngle : 0, outputScale);
		if (!ret)
			return false;
	}
	return true;
}

bool FP(float* D_volumeData, unsigned int volumePitch,
        float* D_projData, unsigned int projPitch,
        const SDimensions& dims, const float* angles,
        const float* TOffsets, float outputScale)
{
	return FP_simple(D_volumeData, volumePitch, D_projData, projPitch,
	                 dims, angles, TOffsets, outputScale);

	// TODO: Fix bug in this non-simple FP with large detscale and TOffsets
#if 0

	// TODO: load angles into constant memory in smaller blocks
	assert(dims.iProjAngles <= g_MaxAngles);

	// TODO: compute region size dynamically to resolve these two assumptions
	// ASSUMPTION: 16 > regionOffset / fDetScale
	const unsigned int g_blockSliceSize = g_detBlockSize;
	assert(16 > (g_blockSlices / g_blockSliceSize) / dims.fDetScale);
	// ASSUMPTION: fDetScale >= 1.0f
	assert(dims.fDetScale > 0.9999f);

	cudaArray* D_dataArray;
	bindVolumeDataTexture(D_volumeData, D_dataArray, volumePitch, dims.iVolWidth, dims.iVolHeight);

	cudaMemcpyToSymbol(gC_angle, angles, dims.iProjAngles*sizeof(float), 0, cudaMemcpyHostToDevice);

	if (TOffsets) {
		cudaMemcpyToSymbol(gC_angle_offset, TOffsets, dims.iProjAngles*sizeof(float), 0, cudaMemcpyHostToDevice);
	} else {
		if (!g_pfZeroes) {
			g_pfZeroes = new float[g_MaxAngles];
			memset(g_pfZeroes, 0, g_MaxAngles * sizeof(float));
		}
		cudaMemcpyToSymbol(gC_angle_offset, g_pfZeroes, dims.iProjAngles*sizeof(float), 0, cudaMemcpyHostToDevice);
	}

	int regionOffset = g_blockSlices / g_blockSliceSize;

	dim3 dimBlock(g_detBlockSize, g_anglesPerBlock); // region size, angles

	std::list<cudaStream_t> streams;


	// Run over all angles, grouping them into groups of the same
	// orientation (roughly horizontal vs. roughly vertical).
	// Start a stream of grids for each such group.

	// TODO: Check if it's worth it to store this info instead
	// of recomputing it every FP.

	unsigned int blockStart = 0;
	unsigned int blockEnd = 0;
	bool blockVertical = false;
	for (unsigned int a = 0; a <= dims.iProjAngles; ++a) {
		bool vertical;
		// TODO: Having <= instead of < below causes a 5% speedup.
		// Maybe we should detect corner cases and put them in the optimal
		// group of angles.
		if (a != dims.iProjAngles)
			vertical = (fabsf(sinf(angles[a])) <= fabsf(cosf(angles[a])));
		if (a == dims.iProjAngles || vertical != blockVertical) {
			// block done

			blockEnd = a;
			if (blockStart != blockEnd) {
				unsigned int length = dims.iVolHeight;
				if (blockVertical)
					length = dims.iVolWidth;
				dim3 dimGrid((blockEnd-blockStart+g_anglesPerBlock-1)/g_anglesPerBlock,
				             (length+g_blockSliceSize-1)/g_blockSliceSize+2*regionOffset); // angle blocks, regions
				// TODO: check if we can't immediately
				//       destroy the stream after use
				cudaStream_t stream;
				cudaStreamCreate(&stream);
				streams.push_back(stream);
				//printf("angle block: %d to %d, %d\n", blockStart, blockEnd, blockVertical);
				if (!blockVertical)
					for (unsigned int i = 0; i < dims.iVolWidth; i += g_blockSlices)
						FPhorizontal<<<dimGrid, dimBlock, 0, stream>>>(D_projData, projPitch, i, blockStart, blockEnd, regionOffset, dims, outputScale);
				else
					for (unsigned int i = 0; i < dims.iVolHeight; i += g_blockSlices)
						FPvertical<<<dimGrid, dimBlock, 0, stream>>>(D_projData, projPitch, i, blockStart, blockEnd, regionOffset, dims, outputScale);
			}
			blockVertical = vertical;
			blockStart = a;
		}
	}

	for (std::list<cudaStream_t>::iterator iter = streams.begin(); iter != streams.end(); ++iter)
		cudaStreamDestroy(*iter);

	streams.clear();

	cudaThreadSynchronize();

	cudaTextForceKernelsCompletion();

	cudaFreeArray(D_dataArray);
		

	return true;
#endif
}


}

#ifdef STANDALONE

using namespace astraCUDA;

int main()
{
	float* D_volumeData;
	float* D_projData;

	SDimensions dims;
	dims.iVolWidth = 1024;
	dims.iVolHeight = 1024;
	dims.iProjAngles = 512;
	dims.iProjDets = 1536;
	dims.fDetScale = 1.0f;
	dims.iRaysPerDet = 1;
	unsigned int volumePitch, projPitch;

	allocateVolume(D_volumeData, dims.iVolWidth, dims.iVolHeight, volumePitch);
	printf("pitch: %u\n", volumePitch);

	allocateVolume(D_projData, dims.iProjDets, dims.iProjAngles, projPitch);
	printf("pitch: %u\n", projPitch);

	unsigned int y, x;
	float* img = loadImage("phantom.png", y, x);

	float* sino = new float[dims.iProjAngles * dims.iProjDets];

	memset(sino, 0, dims.iProjAngles * dims.iProjDets * sizeof(float));

	copyVolumeToDevice(img, dims.iVolWidth, dims.iVolWidth, dims.iVolHeight, D_volumeData, volumePitch);
	copySinogramToDevice(sino, dims.iProjDets, dims.iProjDets, dims.iProjAngles, D_projData, projPitch);

	float* angle = new float[dims.iProjAngles];

	for (unsigned int i = 0; i < dims.iProjAngles; ++i)
		angle[i] = i*(M_PI/dims.iProjAngles);

	FP(D_volumeData, volumePitch, D_projData, projPitch, dims, angle, 0, 1.0f);

	delete[] angle;

	copySinogramFromDevice(sino, dims.iProjDets, dims.iProjDets, dims.iProjAngles, D_projData, projPitch);

	float s = 0.0f;
	for (unsigned int y = 0; y < dims.iProjAngles; ++y)
		for (unsigned int x = 0; x < dims.iProjDets; ++x)
			s += sino[y*dims.iProjDets+x] * sino[y*dims.iProjDets+x];
	printf("cpu norm: %f\n", s);

	//zeroVolume(D_projData, projPitch, dims.iProjDets, dims.iProjAngles);
	s = dotProduct2D(D_projData, projPitch, dims.iProjDets, dims.iProjAngles, 1, 0);
	printf("gpu norm: %f\n", s);

	saveImage("sino.png",dims.iProjAngles,dims.iProjDets,sino);


	return 0;
}
#endif
