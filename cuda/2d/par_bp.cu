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
__constant__ float gC_angle_scale[g_MaxAngles];

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

// TODO: Templated version with/without scale? (Or only the global outputscale)
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
		const float scale = gC_angle_scale[angle];

		const float fT = fX * scaled_cos_theta - fY * scaled_sin_theta + TOffset;
		fVal += tex2D(gT_projTexture, fT, fA) * scale;
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
		const float scale = gC_angle_scale[angle];

		float fT = fX * cos_theta - fY * sin_theta + TOffset;

		for (int iSubX = 0; iSubX < dims.iRaysPerPixelDim; ++iSubX) {
			float fTy = fT;
			fT += fSubStep * cos_theta;
			for (int iSubY = 0; iSubY < dims.iRaysPerPixelDim; ++iSubY) {
				fVal += tex2D(gT_projTexture, fTy, fA) * scale;
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

	// NB: The 'scale' constant in devBP is cancelled out by the SART weighting

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
	float* angle_scale = new float[dims.iProjAngles];

	bindProjDataTexture(D_projData, projPitch, dims.iProjDets, dims.iProjAngles);

	for (unsigned int i = 0; i < dims.iProjAngles; ++i) {
		double d = angles[i].fDetUX * angles[i].fRayY - angles[i].fDetUY * angles[i].fRayX;
		angle_scaled_cos[i] = angles[i].fRayY / d;
		angle_scaled_sin[i] = -angles[i].fRayX / d;
		angle_offset[i] = (angles[i].fDetSY * angles[i].fRayX - angles[i].fDetSX * angles[i].fRayY) / d;
		angle_scale[i] = sqrt(angles[i].fRayX * angles[i].fRayX + angles[i].fRayY * angles[i].fRayY) / abs(d);
	}
	//fprintf(stderr, "outputscale in BP_internal: %f, %f\n", fOutputScale, angle_scale[0]);
	//fprintf(stderr, "ray in BP_internal: %f,%f (length %f)\n", angles[0].fRayX, angles[0].fRayY, sqrt(angles[0].fRayX * angles[0].fRayX + angles[0].fRayY * angles[0].fRayY));

	cudaError_t e1 = cudaMemcpyToSymbol(gC_angle_scaled_sin, angle_scaled_sin, dims.iProjAngles*sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaError_t e2 = cudaMemcpyToSymbol(gC_angle_scaled_cos, angle_scaled_cos, dims.iProjAngles*sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaError_t e3 = cudaMemcpyToSymbol(gC_angle_offset, angle_offset, dims.iProjAngles*sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaError_t e4 = cudaMemcpyToSymbol(gC_angle_scale, angle_scale, dims.iProjAngles*sizeof(float), 0, cudaMemcpyHostToDevice);
	assert(e1 == cudaSuccess);
	assert(e2 == cudaSuccess);
	assert(e3 == cudaSuccess);
	assert(e4 == cudaSuccess);


	delete[] angle_scaled_sin;
	delete[] angle_scaled_cos;
	delete[] angle_offset;
	delete[] angle_scale;

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
	// NB: The adjoint scaling factor from regular BP is cancelled out by the SART weighting
	//fOutputScale *= sqrt(angles[angle].fRayX * angles[angle].fRayX + angles[angle].fRayY * angles[angle].fRayY) / abs(d);

	dim3 dimBlock(g_blockSlices, g_blockSliceSize);
	dim3 dimGrid((dims.iVolWidth+g_blockSlices-1)/g_blockSlices,
	             (dims.iVolHeight+g_blockSliceSize-1)/g_blockSliceSize);

	devBP_SART<<<dimGrid, dimBlock>>>(D_volumeData, volumePitch, angle_offset, angle_scaled_sin, angle_scaled_cos, dims, fOutputScale);
	cudaThreadSynchronize();

	cudaTextForceKernelsCompletion();

	return true;
}


}
