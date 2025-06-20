/*
-----------------------------------------------------------------------
Copyright: 2010-2022, imec Vision Lab, University of Antwerp
           2014-2022, CWI, Amsterdam

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

#include "astra/cuda/gpu_runtime_wrapper.h"

#include "astra/cuda/2d/util.h"
#include "astra/cuda/2d/arith.h"

#include <cstdio>
#include <cassert>
#include <iostream>

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

// projection for angles that are roughly horizontal
// (detector roughly vertical)
__global__ void FanFPhorizontal(float* D_projData, unsigned int projPitch, cudaTextureObject_t tex, unsigned int startSlice, unsigned int startAngle, unsigned int endAngle, const SDimensions dims, float outputScale)
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
			fV += tex2D<float>(tex, fX, fY);
			fY -= alpha;
			fX += 1.0f;
		}

		fVal += fV * fDistCorr;

	}

	projData[angle*projPitch+detector] += fVal;
}


// projection for angles that are roughly vertical
// (detector roughly horizontal)
__global__ void FanFPvertical(float* D_projData, unsigned int projPitch, cudaTextureObject_t tex, unsigned int startSlice, unsigned int startAngle, unsigned int endAngle, const SDimensions dims, float outputScale)
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
			fV += tex2D<float>(tex, fX, fY);
			fX -= alpha;
			fY += 1.0f;
		}

		fVal += fV * fDistCorr;

	}

	projData[angle*projPitch+detector] += fVal;
}

using TransferConstantsBuffer = TransferConstantsBuffer_t<float>;

static bool transferConstants(const SFanProjection *projs, unsigned int nth,
                              TransferConstantsBuffer& buf, cudaStream_t stream)
{
	// transfer angles to constant memory
	float* tmp = &(std::get<0>(buf.d))[0];

	bool ok = checkCuda(cudaStreamWaitEvent(stream, buf.event, 0), "transferConstants wait");

#define TRANSFER_TO_CONSTANT(name) do { for (unsigned int i = 0; i < nth; ++i) tmp[i] = projs[i].f##name ; ok &= checkCuda(cudaMemcpyToSymbolAsync(gC_##name, tmp, nth*sizeof(float), 0, cudaMemcpyHostToDevice, stream), "transferConstants transfer"); } while (0)

	TRANSFER_TO_CONSTANT(SrcX);
	TRANSFER_TO_CONSTANT(SrcY);
	TRANSFER_TO_CONSTANT(DetSX);
	TRANSFER_TO_CONSTANT(DetSY);
	TRANSFER_TO_CONSTANT(DetUX);
	TRANSFER_TO_CONSTANT(DetUY);

#undef TRANSFER_TO_CONSTANT

	ok &= checkCuda(cudaEventRecord(buf.event, stream), "transferConstants event");

	return ok;
}

bool FanFP_internal(float* D_volumeData, unsigned int volumePitch,
           float* D_projData, unsigned int projPitch,
           const SDimensions& dims, const SFanProjection* angles,
           float outputScale, cudaStream_t stream)
{
	assert(dims.iProjAngles <= g_MaxAngles);

	cudaArray* D_dataArray;
	cudaTextureObject_t D_texObj;

	if (!createArrayAndTextureObject2D(D_volumeData, D_dataArray, D_texObj, volumePitch, dims.iVolWidth, dims.iVolHeight))
		return false;

	dim3 dimBlock(g_detBlockSize, g_anglesPerBlock); // region size, angles
	const unsigned int g_blockSliceSize = g_detBlockSize;


	unsigned int blockStart = 0;
	unsigned int blockEnd = dims.iProjAngles;

	dim3 dimGrid((blockEnd-blockStart+g_anglesPerBlock-1)/g_anglesPerBlock,
	             (dims.iProjDets+g_blockSliceSize-1)/g_blockSliceSize); // angle blocks, regions

	for (unsigned int i = 0; i < dims.iVolWidth; i += g_blockSlices)
		FanFPhorizontal<<<dimGrid, dimBlock, 0, stream>>>(D_projData, projPitch, D_texObj, i, blockStart, blockEnd, dims, outputScale);

	for (unsigned int i = 0; i < dims.iVolHeight; i += g_blockSlices)
		FanFPvertical<<<dimGrid, dimBlock, 0, stream>>>(D_projData, projPitch, D_texObj, i, blockStart, blockEnd, dims, outputScale);

	bool ok = true;

	ok &= checkCuda(cudaStreamSynchronize(stream), "fan_fp");

	cudaFreeArray(D_dataArray);
	cudaDestroyTextureObject(D_texObj);

	return ok;
}

bool FanFP(float* D_volumeData, unsigned int volumePitch,
           float* D_projData, unsigned int projPitch,
           const SDimensions& dims, const SFanProjection* angles,
           float outputScale)
{
	TransferConstantsBuffer tcbuf(g_MaxAngles);

	cudaStream_t stream;
	if (!checkCuda(cudaStreamCreate(&stream), "FanFP"))
		return false;

	bool ok = true;

	for (unsigned int iAngle = 0; iAngle < dims.iProjAngles; iAngle += g_MaxAngles) {
		SDimensions subdims = dims;
		unsigned int iEndAngle = iAngle + g_MaxAngles;
		if (iEndAngle >= dims.iProjAngles)
			iEndAngle = dims.iProjAngles;
		subdims.iProjAngles = iEndAngle - iAngle;

		ok &= transferConstants(angles + iAngle, subdims.iProjAngles, tcbuf, stream);
		if (!ok)
			break;

		ok &= FanFP_internal(D_volumeData, volumePitch,
		                         D_projData + iAngle * projPitch, projPitch,
		                         subdims, angles + iAngle,
		                         outputScale, stream);
		if (!ok)
			break;
	}
	ok &= checkCuda(cudaStreamSynchronize(stream), "fan_fp");
	cudaStreamDestroy(stream);
	return ok;
}

}
