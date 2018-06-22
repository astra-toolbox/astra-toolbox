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

#ifdef STANDALONE
#include "testutil.h"
#endif

#include <cstdio>
#include <cassert>
#include <iostream>


typedef texture<float, 2, cudaReadModeElementType> texture2D;

static texture2D gT_projTexture;


namespace astraCUDA {

const unsigned int g_anglesPerBlock = 16;
const unsigned int g_blockSliceSize = 32;
const unsigned int g_blockSlices = 16;

const unsigned int g_MaxAngles = 2560;

__constant__ float gC_angle_scaled_sin[g_MaxAngles];
__constant__ float gC_angle_scaled_cos[g_MaxAngles];
__constant__ float gC_angle_offset[g_MaxAngles];

static bool bindProjDataTexture(float* data, unsigned int pitch, unsigned int width, unsigned int height, cudaTextureAddressMode mode = cudaAddressModeBorder)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	gT_projTexture.addressMode[0] = mode;
	gT_projTexture.addressMode[1] = mode;
	gT_projTexture.filterMode = cudaFilterModeLinear;
	gT_projTexture.normalized = false;

	cudaBindTexture2D(0, gT_projTexture, (const void*)data, channelDesc, width, height, sizeof(float)*pitch);

	// TODO: error value?

	return true;
}

__global__ void devBP(float* D_volData, unsigned int volPitch, unsigned int startAngle, const SDimensions dims, float fOutputScale)
{
	const int relX = threadIdx.x;
	const int relY = threadIdx.y;

	int endAngle = startAngle + g_anglesPerBlock;
	if (endAngle > dims.iProjAngles)
		endAngle = dims.iProjAngles;
	const int X = blockIdx.x * g_blockSlices + relX;
	const int Y = blockIdx.y * g_blockSliceSize + relY;

	if (X >= dims.iVolWidth || Y >= dims.iVolHeight)
		return;

	const float fX = ( X - 0.5f*dims.iVolWidth + 0.5f );
	const float fY = ( Y - 0.5f*dims.iVolHeight + 0.5f );

	float* volData = (float*)D_volData;

	float fVal = 0.0f;
	float fA = startAngle + 0.5f;

	for (int angle = startAngle; angle < endAngle; ++angle)
	{
		const float scaled_cos_theta = gC_angle_scaled_cos[angle];
		const float scaled_sin_theta = gC_angle_scaled_sin[angle];
		const float TOffset = gC_angle_offset[angle];

		const float fT = fX * scaled_cos_theta - fY * scaled_sin_theta + TOffset;
		fVal += tex2D(gT_projTexture, fT, fA);
		fA += 1.0f;
	}

	volData[Y*volPitch+X] += fVal * fOutputScale;
}

// supersampling version
__global__ void devBP_SS(float* D_volData, unsigned int volPitch, unsigned int startAngle, const SDimensions dims, float fOutputScale)
{
	const int relX = threadIdx.x;
	const int relY = threadIdx.y;

	int endAngle = startAngle + g_anglesPerBlock;
	if (endAngle > dims.iProjAngles)
		endAngle = dims.iProjAngles;
	const int X = blockIdx.x * g_blockSlices + relX;
	const int Y = blockIdx.y * g_blockSliceSize + relY;

	if (X >= dims.iVolWidth || Y >= dims.iVolHeight)
		return;

	const float fX = ( X - 0.5f*dims.iVolWidth + 0.5f - 0.5f + 0.5f/dims.iRaysPerPixelDim);
	const float fY = ( Y - 0.5f*dims.iVolHeight + 0.5f - 0.5f + 0.5f/dims.iRaysPerPixelDim);

	const float fSubStep = 1.0f/(dims.iRaysPerPixelDim); // * dims.fDetScale);

	float* volData = (float*)D_volData;

	float fVal = 0.0f;
	float fA = startAngle + 0.5f;

	fOutputScale /= (dims.iRaysPerPixelDim * dims.iRaysPerPixelDim);

	for (int angle = startAngle; angle < endAngle; ++angle)
	{
		const float cos_theta = gC_angle_scaled_cos[angle];
		const float sin_theta = gC_angle_scaled_sin[angle];
		const float TOffset = gC_angle_offset[angle];

		float fT = fX * cos_theta - fY * sin_theta + TOffset;

		for (int iSubX = 0; iSubX < dims.iRaysPerPixelDim; ++iSubX) {
			float fTy = fT;
			fT += fSubStep * cos_theta;
			for (int iSubY = 0; iSubY < dims.iRaysPerPixelDim; ++iSubY) {
				fVal += tex2D(gT_projTexture, fTy, fA);
				fTy -= fSubStep * sin_theta;
			}
		}
		fA += 1.0f;
	}

	volData[Y*volPitch+X] += fVal * fOutputScale;
}

__global__ void devBP_SART(float* D_volData, unsigned int volPitch, float offset, float angle_sin, float angle_cos, const SDimensions dims, float fOutputScale)
{
	const int relX = threadIdx.x;
	const int relY = threadIdx.y;

	const int X = blockIdx.x * g_blockSlices + relX;
	const int Y = blockIdx.y * g_blockSliceSize + relY;

	if (X >= dims.iVolWidth || Y >= dims.iVolHeight)
		return;

	const float fX = ( X - 0.5f*dims.iVolWidth + 0.5f );
	const float fY = ( Y - 0.5f*dims.iVolHeight + 0.5f );

	const float fT = fX * angle_cos - fY * angle_sin + offset;
	const float fVal = tex2D(gT_projTexture, fT, 0.5f);

	D_volData[Y*volPitch+X] += fVal * fOutputScale;
}


bool BP_internal(float* D_volumeData, unsigned int volumePitch,
        float* D_projData, unsigned int projPitch,
        const SDimensions& dims, const SParProjection* angles,
        float fOutputScale)
{
	assert(dims.iProjAngles <= g_MaxAngles);

	float* angle_scaled_sin = new float[dims.iProjAngles];
	float* angle_scaled_cos = new float[dims.iProjAngles];
	float* angle_offset = new float[dims.iProjAngles];

	bindProjDataTexture(D_projData, projPitch, dims.iProjDets, dims.iProjAngles);

	for (unsigned int i = 0; i < dims.iProjAngles; ++i) {
		double d = angles[i].fDetUX * angles[i].fRayY - angles[i].fDetUY * angles[i].fRayX;
		angle_scaled_cos[i] = angles[i].fRayY / d;
		angle_scaled_sin[i] = -angles[i].fRayX / d; // TODO: Check signs
		angle_offset[i] = (angles[i].fDetSY * angles[i].fRayX - angles[i].fDetSX * angles[i].fRayY) / d;
	}

	cudaError_t e1 = cudaMemcpyToSymbol(gC_angle_scaled_sin, angle_scaled_sin, dims.iProjAngles*sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaError_t e2 = cudaMemcpyToSymbol(gC_angle_scaled_cos, angle_scaled_cos, dims.iProjAngles*sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaError_t e3 = cudaMemcpyToSymbol(gC_angle_offset, angle_offset, dims.iProjAngles*sizeof(float), 0, cudaMemcpyHostToDevice);
	assert(e1 == cudaSuccess);
	assert(e2 == cudaSuccess);
	assert(e3 == cudaSuccess);


	delete[] angle_scaled_sin;
	delete[] angle_scaled_cos;
	delete[] angle_offset;

	dim3 dimBlock(g_blockSlices, g_blockSliceSize);
	dim3 dimGrid((dims.iVolWidth+g_blockSlices-1)/g_blockSlices,
	             (dims.iVolHeight+g_blockSliceSize-1)/g_blockSliceSize);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	for (unsigned int i = 0; i < dims.iProjAngles; i += g_anglesPerBlock) {

		if (dims.iRaysPerPixelDim > 1)
			devBP_SS<<<dimGrid, dimBlock, 0, stream>>>(D_volumeData, volumePitch, i, dims, fOutputScale);
		else
			devBP<<<dimGrid, dimBlock, 0, stream>>>(D_volumeData, volumePitch, i, dims, fOutputScale);
	}
	cudaThreadSynchronize();

	cudaTextForceKernelsCompletion();

	cudaStreamDestroy(stream);

	return true;
}

bool BP(float* D_volumeData, unsigned int volumePitch,
        float* D_projData, unsigned int projPitch,
        const SDimensions& dims, const SParProjection* angles, float fOutputScale)
{
	for (unsigned int iAngle = 0; iAngle < dims.iProjAngles; iAngle += g_MaxAngles) {
		SDimensions subdims = dims;
		unsigned int iEndAngle = iAngle + g_MaxAngles;
		if (iEndAngle >= dims.iProjAngles)
			iEndAngle = dims.iProjAngles;
		subdims.iProjAngles = iEndAngle - iAngle;

		bool ret;
		ret = BP_internal(D_volumeData, volumePitch,
		                  D_projData + iAngle * projPitch, projPitch,
		                  subdims, angles + iAngle, fOutputScale);
		if (!ret)
			return false;
	}
	return true;
}


bool BP_SART(float* D_volumeData, unsigned int volumePitch,
             float* D_projData, unsigned int projPitch,
             unsigned int angle, const SDimensions& dims,
             const SParProjection* angles, float fOutputScale)
{
	// Only one angle.
	// We need to Clamp to the border pixels instead of to zero, because
	// SART weights with ray length.
	bindProjDataTexture(D_projData, projPitch, dims.iProjDets, 1, cudaAddressModeClamp);

	double d = angles[angle].fDetUX * angles[angle].fRayY - angles[angle].fDetUY * angles[angle].fRayX;
	float angle_scaled_cos = angles[angle].fRayY / d;
	float angle_scaled_sin = -angles[angle].fRayX / d; // TODO: Check signs
	float angle_offset = (angles[angle].fDetSY * angles[angle].fRayX - angles[angle].fDetSX * angles[angle].fRayY) / d;

	dim3 dimBlock(g_blockSlices, g_blockSliceSize);
	dim3 dimGrid((dims.iVolWidth+g_blockSlices-1)/g_blockSlices,
	             (dims.iVolHeight+g_blockSliceSize-1)/g_blockSliceSize);

	devBP_SART<<<dimGrid, dimBlock>>>(D_volumeData, volumePitch, angle_offset, angle_scaled_sin, angle_scaled_cos, dims, fOutputScale);
	cudaThreadSynchronize();

	cudaTextForceKernelsCompletion();

	return true;
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
	float* sino = loadImage("sino.png", y, x);

	float* img = new float[dims.iVolWidth*dims.iVolHeight];

	memset(img, 0, dims.iVolWidth*dims.iVolHeight*sizeof(float));

	copyVolumeToDevice(img, dims.iVolWidth, dims.iVolWidth, dims.iVolHeight, D_volumeData, volumePitch);
	copySinogramToDevice(sino, dims.iProjDets, dims.iProjDets, dims.iProjAngles, D_projData, projPitch);

	float* angle = new float[dims.iProjAngles];

	for (unsigned int i = 0; i < dims.iProjAngles; ++i)
		angle[i] = i*(M_PI/dims.iProjAngles);

	BP(D_volumeData, volumePitch, D_projData, projPitch, dims, angle, 0, 1.0f);

	delete[] angle;

	copyVolumeFromDevice(img, dims.iVolWidth, dims.iVolWidth, dims.iVolHeight, D_volumeData, volumePitch);

	saveImage("vol.png",dims.iVolHeight,dims.iVolWidth,img);

	return 0;
}
#endif
