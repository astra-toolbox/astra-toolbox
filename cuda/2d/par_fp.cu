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

#include "astra/cuda/2d/util.h"
#include "astra/cuda/2d/arith.h"

#include <cstdio>
#include <cassert>
#include <iostream>
#include <list>
#include <cmath>

typedef texture<float, 2, cudaReadModeElementType> texture2D;

static texture2D gT_volumeTexture;


namespace astraCUDA {

static const unsigned g_MaxAngles = 2560;
__constant__ float gC_angle[g_MaxAngles];
__constant__ float gC_angle_offset[g_MaxAngles];
__constant__ float gC_angle_detsize[g_MaxAngles];


// optimization parameters
static const unsigned int g_anglesPerBlock = 16;
static const unsigned int g_detBlockSize = 32;
static const unsigned int g_blockSlices = 64;

// fixed point scaling factor
#define fPREC_FACTOR 16.0f
#define iPREC_FACTOR 16


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

	const float fDetStep = -gC_angle_detsize[angle] / sin_theta;
	float fSliceStep = cos_theta / sin_theta;
	float fDistCorr;
	if (sin_theta > 0.0f)
		fDistCorr = outputScale / sin_theta;
	else
		fDistCorr = -outputScale / sin_theta;

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

	const float fDetStep = gC_angle_detsize[angle] / cos_theta;
	float fSliceStep = sin_theta / cos_theta;
	float fDistCorr;
	if (cos_theta < 0.0f)
		fDistCorr = -outputScale / cos_theta; 
	else
		fDistCorr = outputScale / cos_theta;

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




// Coordinates of center of detector pixel number t:
// x = (t - 0.5*nDets + 0.5 - fOffset) * fSize * cos(fAngle)
// y = - (t - 0.5*nDets + 0.5 - fOffset) * fSize * sin(fAngle)


static void convertAndUploadAngles(const SParProjection *projs, unsigned int nth, unsigned int ndets)
{
	float *angles = new float[nth];
	float *offsets = new float[nth];
	float *detsizes = new float[nth];

	for (int i = 0; i < nth; ++i)
		getParParameters(projs[i], ndets, angles[i], detsizes[i], offsets[i]);

	cudaMemcpyToSymbol(gC_angle, angles, nth*sizeof(float), 0, cudaMemcpyHostToDevice); 
	cudaMemcpyToSymbol(gC_angle_offset, offsets, nth*sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(gC_angle_detsize, detsizes, nth*sizeof(float), 0, cudaMemcpyHostToDevice); 
}



bool FP_simple_internal(float* D_volumeData, unsigned int volumePitch,
               float* D_projData, unsigned int projPitch,
               const SDimensions& dims, const SParProjection* angles,
               float outputScale)
{
	assert(dims.iProjAngles <= g_MaxAngles);

	assert(angles);

	cudaArray* D_dataArray;
	bindVolumeDataTexture(D_volumeData, D_dataArray, volumePitch, dims.iVolWidth, dims.iVolHeight);


	convertAndUploadAngles(angles, dims.iProjAngles, dims.iProjDets);


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
			vertical = (fabsf(angles[a].fRayX) <= fabsf(angles[a].fRayY));
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

	cudaDeviceSynchronize();

	cudaTextForceKernelsCompletion();

	cudaFreeArray(D_dataArray);
		

	return true;
}

bool FP_simple(float* D_volumeData, unsigned int volumePitch,
               float* D_projData, unsigned int projPitch,
               const SDimensions& dims, const SParProjection* angles,
               float outputScale)
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
		                         outputScale);
		if (!ret)
			return false;
	}
	return true;
}

bool FP(float* D_volumeData, unsigned int volumePitch,
        float* D_projData, unsigned int projPitch,
        const SDimensions& dims, const SParProjection* angles,
        float outputScale)
{
	return FP_simple(D_volumeData, volumePitch, D_projData, projPitch,
	                 dims, angles, outputScale);

}


}
