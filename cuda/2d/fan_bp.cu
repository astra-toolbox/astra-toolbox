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

#include "util.h"
#include "arith.h"

#ifdef STANDALONE
#include "testutil.h"
#endif

#define PIXELTRACE


typedef texture<float, 2, cudaReadModeElementType> texture2D;

static texture2D gT_FanProjTexture;


namespace astraCUDA {

const unsigned int g_anglesPerBlock = 16;
const unsigned int g_blockSliceSize = 32;
const unsigned int g_blockSlices = 16;

const unsigned int g_MaxAngles = 2560;

__constant__ float gC_SrcX[g_MaxAngles];
__constant__ float gC_SrcY[g_MaxAngles];
__constant__ float gC_DetSX[g_MaxAngles];
__constant__ float gC_DetSY[g_MaxAngles];
__constant__ float gC_DetUX[g_MaxAngles];
__constant__ float gC_DetUY[g_MaxAngles];


static bool bindProjDataTexture(float* data, unsigned int pitch, unsigned int width, unsigned int height, cudaTextureAddressMode mode = cudaAddressModeBorder)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	gT_FanProjTexture.addressMode[0] = mode;
	gT_FanProjTexture.addressMode[1] = mode;
	gT_FanProjTexture.filterMode = cudaFilterModeLinear;
	gT_FanProjTexture.normalized = false;

	cudaBindTexture2D(0, gT_FanProjTexture, (const void*)data, channelDesc, width, height, sizeof(float)*pitch);

	// TODO: error value?

	return true;
}

__global__ void devFanBP(float* D_volData, unsigned int volPitch, unsigned int startAngle, const SDimensions dims, float fOutputScale)
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
	const float fY = - ( Y - 0.5f*dims.iVolHeight + 0.5f );

	float* volData = (float*)D_volData;

	float fVal = 0.0f;
	float fA = startAngle + 0.5f;

	// TODO: Distance correction?

	for (int angle = startAngle; angle < endAngle; ++angle)
	{
		const float fSrcX = gC_SrcX[angle];
		const float fSrcY = gC_SrcY[angle];
		const float fDetSX = gC_DetSX[angle];
		const float fDetSY = gC_DetSY[angle];
		const float fDetUX = gC_DetUX[angle];
		const float fDetUY = gC_DetUY[angle];

		const float fXD = fSrcX - fX;
		const float fYD = fSrcY - fY;

		const float fNum = fDetSY * fXD - fDetSX * fYD + fX*fSrcY - fY*fSrcX;
		const float fDen = fDetUX * fYD - fDetUY * fXD;
		
		const float fT = fNum / fDen;
		fVal += tex2D(gT_FanProjTexture, fT, fA);
		fA += 1.0f;
	}

	volData[Y*volPitch+X] += fVal * fOutputScale;
}

// supersampling version
__global__ void devFanBP_SS(float* D_volData, unsigned int volPitch, unsigned int startAngle, const SDimensions dims, float fOutputScale)
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

	const float fXb = ( X - 0.5f*dims.iVolWidth + 0.5f - 0.5f + 0.5f/dims.iRaysPerPixelDim);
	const float fYb = - ( Y - 0.5f*dims.iVolHeight + 0.5f - 0.5f + 0.5f/dims.iRaysPerPixelDim);

	const float fSubStep = 1.0f/dims.iRaysPerPixelDim;

	float* volData = (float*)D_volData;

	fOutputScale /= (dims.iRaysPerPixelDim * dims.iRaysPerPixelDim);

	float fVal = 0.0f;
	float fA = startAngle + 0.5f;

	// TODO: Distance correction?

	for (int angle = startAngle; angle < endAngle; ++angle)
	{
		const float fSrcX = gC_SrcX[angle];
		const float fSrcY = gC_SrcY[angle];
		const float fDetSX = gC_DetSX[angle];
		const float fDetSY = gC_DetSY[angle];
		const float fDetUX = gC_DetUX[angle];
		const float fDetUY = gC_DetUY[angle];

		// TODO: Optimize these loops...
		float fX = fXb;
		for (int iSubX = 0; iSubX < dims.iRaysPerPixelDim; ++iSubX) {
			float fY = fYb;
			for (int iSubY = 0; iSubY < dims.iRaysPerPixelDim; ++iSubY) {
				const float fXD = fSrcX - fX;
				const float fYD = fSrcY - fY;

				const float fNum = fDetSY * fXD - fDetSX * fYD + fX*fSrcY - fY*fSrcX;
				const float fDen = fDetUX * fYD - fDetUY * fXD;
		
				const float fT = fNum / fDen;
				fVal += tex2D(gT_FanProjTexture, fT, fA);
				fY -= fSubStep;
			}
			fX += fSubStep;
		}
		fA += 1.0f;
	}

	volData[Y*volPitch+X] += fVal * fOutputScale;
}


// BP specifically for SART.
// It includes (free) weighting with voxel weight.
// It assumes the proj texture is set up _without_ padding, unlike regular BP.
__global__ void devFanBP_SART(float* D_volData, unsigned int volPitch, const SDimensions dims, float fOutputScale)
{
	const int relX = threadIdx.x;
	const int relY = threadIdx.y;

	const int X = blockIdx.x * g_blockSlices + relX;
	const int Y = blockIdx.y * g_blockSliceSize + relY;

	if (X >= dims.iVolWidth || Y >= dims.iVolHeight)
		return;

	const float fX = ( X - 0.5f*dims.iVolWidth + 0.5f );
	const float fY = - ( Y - 0.5f*dims.iVolHeight + 0.5f );

	float* volData = (float*)D_volData;

	// TODO: Distance correction?

	// TODO: Constant memory vs parameters.
	const float fSrcX = gC_SrcX[0];
	const float fSrcY = gC_SrcY[0];
	const float fDetSX = gC_DetSX[0];
	const float fDetSY = gC_DetSY[0];
	const float fDetUX = gC_DetUX[0];
	const float fDetUY = gC_DetUY[0];

	const float fXD = fSrcX - fX;
	const float fYD = fSrcY - fY;

	const float fNum = fDetSY * fXD - fDetSX * fYD + fX*fSrcY - fY*fSrcX;
	const float fDen = fDetUX * fYD - fDetUY * fXD;
		
	const float fT = fNum / fDen;
	const float fVal = tex2D(gT_FanProjTexture, fT, 0.5f);

	volData[Y*volPitch+X] += fVal * fOutputScale;
}

// Weighted BP for use in fan beam FBP
// Each pixel/ray is weighted by 1/L^2 where L is the distance to the source.
__global__ void devFanBP_FBPWeighted(float* D_volData, unsigned int volPitch, unsigned int startAngle, const SDimensions dims, float fOutputScale)
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
	const float fY = - ( Y - 0.5f*dims.iVolHeight + 0.5f );

	float* volData = (float*)D_volData;

	float fVal = 0.0f;
	float fA = startAngle + 0.5f;

	// TODO: Distance correction?

	for (int angle = startAngle; angle < endAngle; ++angle)
	{
		const float fSrcX = gC_SrcX[angle];
		const float fSrcY = gC_SrcY[angle];
		const float fDetSX = gC_DetSX[angle];
		const float fDetSY = gC_DetSY[angle];
		const float fDetUX = gC_DetUX[angle];
		const float fDetUY = gC_DetUY[angle];

		const float fXD = fSrcX - fX;
		const float fYD = fSrcY - fY;

		const float fNum = fDetSY * fXD - fDetSX * fYD + fX*fSrcY - fY*fSrcX;
		const float fDen = fDetUX * fYD - fDetUY * fXD;

		const float fWeight = fXD*fXD + fYD*fYD;
		
		const float fT = fNum / fDen;
		fVal += tex2D(gT_FanProjTexture, fT, fA) / fWeight;
		fA += 1.0f;
	}

	volData[Y*volPitch+X] += fVal * fOutputScale;
}


bool FanBP_internal(float* D_volumeData, unsigned int volumePitch,
           float* D_projData, unsigned int projPitch,
           const SDimensions& dims, const SFanProjection* angles,
           float fOutputScale)
{
	assert(dims.iProjAngles <= g_MaxAngles);

	bindProjDataTexture(D_projData, projPitch, dims.iProjDets, dims.iProjAngles);

	// transfer angles to constant memory
	float* tmp = new float[dims.iProjAngles];

#define TRANSFER_TO_CONSTANT(name) do { for (unsigned int i = 0; i < dims.iProjAngles; ++i) tmp[i] = angles[i].f##name ; cudaMemcpyToSymbol(gC_##name, tmp, dims.iProjAngles*sizeof(float), 0, cudaMemcpyHostToDevice); } while (0)

	TRANSFER_TO_CONSTANT(SrcX);
	TRANSFER_TO_CONSTANT(SrcY);
	TRANSFER_TO_CONSTANT(DetSX);
	TRANSFER_TO_CONSTANT(DetSY);
	TRANSFER_TO_CONSTANT(DetUX);
	TRANSFER_TO_CONSTANT(DetUY);

#undef TRANSFER_TO_CONSTANT

	delete[] tmp;

	dim3 dimBlock(g_blockSlices, g_blockSliceSize);
	dim3 dimGrid((dims.iVolWidth+g_blockSlices-1)/g_blockSlices,
	             (dims.iVolHeight+g_blockSliceSize-1)/g_blockSliceSize);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	for (unsigned int i = 0; i < dims.iProjAngles; i += g_anglesPerBlock) {
		if (dims.iRaysPerPixelDim > 1)
			devFanBP_SS<<<dimGrid, dimBlock, 0, stream>>>(D_volumeData, volumePitch, i, dims, fOutputScale);
		else
			devFanBP<<<dimGrid, dimBlock, 0, stream>>>(D_volumeData, volumePitch, i, dims, fOutputScale);
	}
	cudaThreadSynchronize();

	cudaTextForceKernelsCompletion();

	cudaStreamDestroy(stream);

	return true;
}

bool FanBP_FBPWeighted_internal(float* D_volumeData, unsigned int volumePitch,
           float* D_projData, unsigned int projPitch,
           const SDimensions& dims, const SFanProjection* angles,
           float fOutputScale)
{
	assert(dims.iProjAngles <= g_MaxAngles);

	bindProjDataTexture(D_projData, projPitch, dims.iProjDets, dims.iProjAngles);

	// transfer angles to constant memory
	float* tmp = new float[dims.iProjAngles];

#define TRANSFER_TO_CONSTANT(name) do { for (unsigned int i = 0; i < dims.iProjAngles; ++i) tmp[i] = angles[i].f##name ; cudaMemcpyToSymbol(gC_##name, tmp, dims.iProjAngles*sizeof(float), 0, cudaMemcpyHostToDevice); } while (0)

	TRANSFER_TO_CONSTANT(SrcX);
	TRANSFER_TO_CONSTANT(SrcY);
	TRANSFER_TO_CONSTANT(DetSX);
	TRANSFER_TO_CONSTANT(DetSY);
	TRANSFER_TO_CONSTANT(DetUX);
	TRANSFER_TO_CONSTANT(DetUY);

#undef TRANSFER_TO_CONSTANT

	delete[] tmp;

	dim3 dimBlock(g_blockSlices, g_blockSliceSize);
	dim3 dimGrid((dims.iVolWidth+g_blockSlices-1)/g_blockSlices,
	             (dims.iVolHeight+g_blockSliceSize-1)/g_blockSliceSize);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	for (unsigned int i = 0; i < dims.iProjAngles; i += g_anglesPerBlock) {
		devFanBP_FBPWeighted<<<dimGrid, dimBlock, 0, stream>>>(D_volumeData, volumePitch, i, dims, fOutputScale);
	}
	cudaThreadSynchronize();

	cudaTextForceKernelsCompletion();

	cudaStreamDestroy(stream);

	return true;
}

// D_projData is a pointer to one padded sinogram line
bool FanBP_SART(float* D_volumeData, unsigned int volumePitch,
                float* D_projData, unsigned int projPitch,
                unsigned int angle,
                const SDimensions& dims, const SFanProjection* angles,
                float fOutputScale)
{
	// only one angle
	bindProjDataTexture(D_projData, projPitch, dims.iProjDets, 1, cudaAddressModeClamp);

	// transfer angle to constant memory
#define TRANSFER_TO_CONSTANT(name) do { cudaMemcpyToSymbol(gC_##name, &(angles[angle].f##name), sizeof(float), 0, cudaMemcpyHostToDevice); } while (0)

	TRANSFER_TO_CONSTANT(SrcX);
	TRANSFER_TO_CONSTANT(SrcY);
	TRANSFER_TO_CONSTANT(DetSX);
	TRANSFER_TO_CONSTANT(DetSY);
	TRANSFER_TO_CONSTANT(DetUX);
	TRANSFER_TO_CONSTANT(DetUY);

#undef TRANSFER_TO_CONSTANT

	dim3 dimBlock(g_blockSlices, g_blockSliceSize);
	dim3 dimGrid((dims.iVolWidth+g_blockSlices-1)/g_blockSlices,
	             (dims.iVolHeight+g_blockSliceSize-1)/g_blockSliceSize);

	devFanBP_SART<<<dimGrid, dimBlock>>>(D_volumeData, volumePitch, dims, fOutputScale);
	cudaThreadSynchronize();

	cudaTextForceKernelsCompletion();

	return true;
}

bool FanBP(float* D_volumeData, unsigned int volumePitch,
           float* D_projData, unsigned int projPitch,
           const SDimensions& dims, const SFanProjection* angles,
           float fOutputScale)
{
	for (unsigned int iAngle = 0; iAngle < dims.iProjAngles; iAngle += g_MaxAngles) {
		SDimensions subdims = dims;
		unsigned int iEndAngle = iAngle + g_MaxAngles;
		if (iEndAngle >= dims.iProjAngles)
			iEndAngle = dims.iProjAngles;
		subdims.iProjAngles = iEndAngle - iAngle;

		bool ret;
		ret = FanBP_internal(D_volumeData, volumePitch,
		                  D_projData + iAngle * projPitch, projPitch,
		                  subdims, angles + iAngle, fOutputScale);
		if (!ret)
			return false;
	}
	return true;
}

bool FanBP_FBPWeighted(float* D_volumeData, unsigned int volumePitch,
           float* D_projData, unsigned int projPitch,
           const SDimensions& dims, const SFanProjection* angles,
           float fOutputScale)
{
	for (unsigned int iAngle = 0; iAngle < dims.iProjAngles; iAngle += g_MaxAngles) {
		SDimensions subdims = dims;
		unsigned int iEndAngle = iAngle + g_MaxAngles;
		if (iEndAngle >= dims.iProjAngles)
			iEndAngle = dims.iProjAngles;
		subdims.iProjAngles = iEndAngle - iAngle;

		bool ret;
		ret = FanBP_FBPWeighted_internal(D_volumeData, volumePitch,
		                  D_projData + iAngle * projPitch, projPitch,
		                  subdims, angles + iAngle, fOutputScale);

		if (!ret)
			return false;
	}
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
	dims.iVolWidth = 128;
	dims.iVolHeight = 128;
	dims.iProjAngles = 180;
	dims.iProjDets = 256;
	dims.fDetScale = 1.0f;
	dims.iRaysPerDet = 1;
	unsigned int volumePitch, projPitch;

	SFanProjection projs[180];

	projs[0].fSrcX = 0.0f;
	projs[0].fSrcY = 1536.0f;
	projs[0].fDetSX = 128.0f;
	projs[0].fDetSY = -512.0f;
	projs[0].fDetUX = -1.0f;
	projs[0].fDetUY = 0.0f;

#define ROTATE0(name,i,alpha) do { projs[i].f##name##X = projs[0].f##name##X * cos(alpha) - projs[0].f##name##Y * sin(alpha); projs[i].f##name##Y = projs[0].f##name##X * sin(alpha) + projs[0].f##name##Y * cos(alpha); } while(0)

	for (int i = 1; i < 180; ++i) {
		ROTATE0(Src, i, i*2*M_PI/180);
		ROTATE0(DetS, i, i*2*M_PI/180);
		ROTATE0(DetU, i, i*2*M_PI/180);
	}

#undef ROTATE0

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

	FanBP(D_volumeData, volumePitch, D_projData, projPitch, dims, projs, 1.0f);

	copyVolumeFromDevice(img, dims.iVolWidth, dims.iVolWidth, dims.iVolHeight, D_volumeData, volumePitch);

	saveImage("vol.png",dims.iVolHeight,dims.iVolWidth,img);

	return 0;
}
#endif
