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


typedef texture<float, 2, cudaReadModeElementType> texture2D;

static texture2D gT_FanVolumeTexture;


namespace astraCUDA {

static const unsigned g_MaxAngles = 2560;
__constant__ float gC_SrcX[g_MaxAngles];
__constant__ float gC_SrcY[g_MaxAngles];
__constant__ float gC_DetSX[g_MaxAngles];
__constant__ float gC_DetSY[g_MaxAngles];
__constant__ float gC_DetUX[g_MaxAngles];
__constant__ float gC_DetUY[g_MaxAngles];


// optimization parameters
static const unsigned int g_anglesPerBlock = 16;
static const unsigned int g_detBlockSize = 32;
static const unsigned int g_blockSlices = 64;

static bool bindVolumeDataTexture(float* data, cudaArray*& dataArray, unsigned int pitch, unsigned int width, unsigned int height)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	dataArray = 0;
	cudaMallocArray(&dataArray, &channelDesc, width, height);
	cudaMemcpy2DToArray(dataArray, 0, 0, data, pitch*sizeof(float), width*sizeof(float), height, cudaMemcpyDeviceToDevice);

	gT_FanVolumeTexture.addressMode[0] = cudaAddressModeBorder;
	gT_FanVolumeTexture.addressMode[1] = cudaAddressModeBorder;
	gT_FanVolumeTexture.filterMode = cudaFilterModeLinear;
	gT_FanVolumeTexture.normalized = false;

	// TODO: For very small sizes (roughly <=512x128) with few angles (<=180)
	// not using an array is more efficient.
	//cudaBindTexture2D(0, gT_FanVolumeTexture, (const void*)data, channelDesc, width, height, sizeof(float)*pitch);
	cudaBindTextureToArray(gT_FanVolumeTexture, dataArray, channelDesc);

	// TODO: error value?

	return true;
}

// projection for angles that are roughly horizontal
// (detector roughly vertical)
__global__ void FanFPhorizontal(float* D_projData, unsigned int projPitch, unsigned int startSlice, unsigned int startAngle, unsigned int endAngle, const SDimensions dims, float outputScale)
{
	float* projData = (float*)D_projData;
	const int relDet = threadIdx.x;
	const int relAngle = threadIdx.y;

	const int angle = startAngle + blockIdx.x * g_anglesPerBlock + relAngle;
	if (angle >= endAngle)
		return;

	const int detector = blockIdx.y * g_detBlockSize + relDet;

	if (detector < 0 || detector >= dims.iProjDets)
		return;

	const float fSrcX = gC_SrcX[angle];
	const float fSrcY = gC_SrcY[angle];
	const float fDetSX = gC_DetSX[angle];
	const float fDetSY = gC_DetSY[angle];
	const float fDetUX = gC_DetUX[angle];
	const float fDetUY = gC_DetUY[angle];

	float fVal = 0.0f;

	const float fdx = fabsf(fDetSX + detector*fDetUX + 0.5f - fSrcX);
	const float fdy = fabsf(fDetSY + detector*fDetUY + 0.5f - fSrcY);

	if (fdy > fdx)
		return;


	for (int iSubT = 0; iSubT < dims.iRaysPerDet; ++iSubT) {
		const float fDet = detector + (0.5f + iSubT) / dims.iRaysPerDet;

		const float fDetX = fDetSX + fDet * fDetUX;
		const float fDetY = fDetSY + fDet * fDetUY;

		// ray: y = alpha * x + beta
		const float alpha = (fSrcY - fDetY) / (fSrcX - fDetX);
		const float beta = fSrcY - alpha * fSrcX;
	
		const float fDistCorr = sqrt(alpha*alpha+1.0f) * outputScale / dims.iRaysPerDet;

		// intersect ray with first slice

		float fY = -alpha * (startSlice - 0.5f*dims.iVolWidth + 0.5f) - beta + 0.5f*dims.iVolHeight - 0.5f + 0.5f;
		float fX = startSlice + 0.5f;

		int endSlice = startSlice + g_blockSlices;
		if (endSlice > dims.iVolWidth)
			endSlice = dims.iVolWidth;

		float fV = 0.0f;
		for (int slice = startSlice; slice < endSlice; ++slice)
		{
			fV += tex2D(gT_FanVolumeTexture, fX, fY);
			fY -= alpha;
			fX += 1.0f;
		}

		fVal += fV * fDistCorr;

	}

	projData[angle*projPitch+detector] += fVal;
}


// projection for angles that are roughly vertical
// (detector roughly horizontal)
__global__ void FanFPvertical(float* D_projData, unsigned int projPitch, unsigned int startSlice, unsigned int startAngle, unsigned int endAngle, const SDimensions dims, float outputScale)
{
	const int relDet = threadIdx.x;
	const int relAngle = threadIdx.y;

	const int angle = startAngle + blockIdx.x * g_anglesPerBlock + relAngle;

	if (angle >= endAngle)
		return;

	const int detector = blockIdx.y * g_detBlockSize + relDet;

	if (detector < 0 || detector >= dims.iProjDets)
		return;

	float* projData = (float*)D_projData;

	const float fSrcX = gC_SrcX[angle];
	const float fSrcY = gC_SrcY[angle];
	const float fDetSX = gC_DetSX[angle];
	const float fDetSY = gC_DetSY[angle];
	const float fDetUX = gC_DetUX[angle];
	const float fDetUY = gC_DetUY[angle];

	float fVal = 0.0f;

	const float fdx = fabsf(fDetSX + detector*fDetUX + 0.5f - fSrcX);
	const float fdy = fabsf(fDetSY + detector*fDetUY + 0.5f - fSrcY);

	if (fdy <= fdx)
		return;


	for (int iSubT = 0; iSubT < dims.iRaysPerDet; ++iSubT) {
		const float fDet = detector + (0.5f + iSubT) / dims.iRaysPerDet /*- gC_angle_offset[angle]*/;

		const float fDetX = fDetSX + fDet * fDetUX;
		const float fDetY = fDetSY + fDet * fDetUY;

		// ray: x = alpha * y + beta
		const float alpha = (fSrcX - fDetX) / (fSrcY - fDetY);
		const float beta = fSrcX - alpha * fSrcY;
	
		const float fDistCorr = sqrt(alpha*alpha+1) * outputScale / dims.iRaysPerDet;

		// intersect ray with first slice

		float fX = -alpha * (startSlice - 0.5f*dims.iVolHeight + 0.5f) + beta + 0.5f*dims.iVolWidth - 0.5f + 0.5f;
		float fY = startSlice + 0.5f;

		int endSlice = startSlice + g_blockSlices;
		if (endSlice > dims.iVolHeight)
			endSlice = dims.iVolHeight;

		float fV = 0.0f;

		for (int slice = startSlice; slice < endSlice; ++slice)
		{
			fV += tex2D(gT_FanVolumeTexture, fX, fY);
			fX -= alpha;
			fY += 1.0f;
		}

		fVal += fV * fDistCorr;

	}

	projData[angle*projPitch+detector] += fVal;
}

bool FanFP_internal(float* D_volumeData, unsigned int volumePitch,
           float* D_projData, unsigned int projPitch,
           const SDimensions& dims, const SFanProjection* angles,
           float outputScale)
{
	assert(dims.iProjAngles <= g_MaxAngles);

	cudaArray* D_dataArray;
	bindVolumeDataTexture(D_volumeData, D_dataArray, volumePitch, dims.iVolWidth, dims.iVolHeight);

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

	dim3 dimBlock(g_detBlockSize, g_anglesPerBlock); // region size, angles
	const unsigned int g_blockSliceSize = g_detBlockSize;

	std::list<cudaStream_t> streams;


	unsigned int blockStart = 0;
	unsigned int blockEnd = dims.iProjAngles;

	dim3 dimGrid((blockEnd-blockStart+g_anglesPerBlock-1)/g_anglesPerBlock,
	             (dims.iProjDets+g_blockSliceSize-1)/g_blockSliceSize); // angle blocks, regions
	cudaStream_t stream1;
	cudaStreamCreate(&stream1);
	streams.push_back(stream1);
	for (unsigned int i = 0; i < dims.iVolWidth; i += g_blockSlices)
		FanFPhorizontal<<<dimGrid, dimBlock, 0, stream1>>>(D_projData, projPitch, i, blockStart, blockEnd, dims, outputScale);

	cudaStream_t stream2;
	cudaStreamCreate(&stream2);
	streams.push_back(stream2);
	for (unsigned int i = 0; i < dims.iVolHeight; i += g_blockSlices)
		FanFPvertical<<<dimGrid, dimBlock, 0, stream2>>>(D_projData, projPitch, i, blockStart, blockEnd, dims, outputScale);

	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);

	cudaThreadSynchronize();

	cudaTextForceKernelsCompletion();

	cudaFreeArray(D_dataArray);

	return true;
}

bool FanFP(float* D_volumeData, unsigned int volumePitch,
           float* D_projData, unsigned int projPitch,
           const SDimensions& dims, const SFanProjection* angles,
           float outputScale)
{
	for (unsigned int iAngle = 0; iAngle < dims.iProjAngles; iAngle += g_MaxAngles) {
		SDimensions subdims = dims;
		unsigned int iEndAngle = iAngle + g_MaxAngles;
		if (iEndAngle >= dims.iProjAngles)
			iEndAngle = dims.iProjAngles;
		subdims.iProjAngles = iEndAngle - iAngle;

		bool ret;
		ret = FanFP_internal(D_volumeData, volumePitch,
		                         D_projData + iAngle * projPitch, projPitch,
		                         subdims, angles + iAngle,
		                         outputScale);
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
	float* img = loadImage("phantom128.png", y, x);

	float* sino = new float[dims.iProjAngles * dims.iProjDets];

	memset(sino, 0, dims.iProjAngles * dims.iProjDets * sizeof(float));

	copyVolumeToDevice(img, dims.iVolWidth, dims.iVolWidth, dims.iVolHeight, D_volumeData, volumePitch);
	copySinogramToDevice(sino, dims.iProjDets, dims.iProjDets, dims.iProjAngles, D_projData, projPitch);

	float* angle = new float[dims.iProjAngles];

	for (unsigned int i = 0; i < dims.iProjAngles; ++i)
		angle[i] = i*(M_PI/dims.iProjAngles);

	FanFP(D_volumeData, volumePitch, D_projData, projPitch, dims, projs, 1.0f);

	delete[] angle;

	copySinogramFromDevice(sino, dims.iProjDets, dims.iProjDets, dims.iProjAngles, D_projData, projPitch);

	float s = 0.0f;
	for (unsigned int y = 0; y < dims.iProjAngles; ++y)
		for (unsigned int x = 0; x < dims.iProjDets; ++x)
			s += sino[y*dims.iProjDets+x] * sino[y*dims.iProjDets+x];
	printf("cpu norm: %f\n", s);

	//zeroVolume(D_projData, projPitch, dims.iProjDets, dims.iProjAngles);
	s = dotProduct2D(D_projData, projPitch, dims.iProjDets, dims.iProjAngles);
	printf("gpu norm: %f\n", s);

	saveImage("sino.png",dims.iProjAngles,dims.iProjDets,sino);


	return 0;
}
#endif
