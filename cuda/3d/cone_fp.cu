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
#include <iostream>
#include <list>

#include <cuda.h>
#include "util3d.h"

#ifdef STANDALONE
#include "testutil.h"
#endif

#include "dims3d.h"

typedef texture<float, 3, cudaReadModeElementType> texture3D;

static texture3D gT_coneVolumeTexture;

namespace astraCUDA3d {

static const unsigned int g_anglesPerBlock = 4;

// thickness of the slices we're splitting the volume up into
static const unsigned int g_blockSlices = 64;
static const unsigned int g_detBlockU = 32;
static const unsigned int g_detBlockV = 32;

static const unsigned g_MaxAngles = 1024;
__constant__ float gC_SrcX[g_MaxAngles];
__constant__ float gC_SrcY[g_MaxAngles];
__constant__ float gC_SrcZ[g_MaxAngles];
__constant__ float gC_DetSX[g_MaxAngles];
__constant__ float gC_DetSY[g_MaxAngles];
__constant__ float gC_DetSZ[g_MaxAngles];
__constant__ float gC_DetUX[g_MaxAngles];
__constant__ float gC_DetUY[g_MaxAngles];
__constant__ float gC_DetUZ[g_MaxAngles];
__constant__ float gC_DetVX[g_MaxAngles];
__constant__ float gC_DetVY[g_MaxAngles];
__constant__ float gC_DetVZ[g_MaxAngles];


bool bindVolumeDataTexture(const cudaArray* array)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	gT_coneVolumeTexture.addressMode[0] = cudaAddressModeBorder;
	gT_coneVolumeTexture.addressMode[1] = cudaAddressModeBorder;
	gT_coneVolumeTexture.addressMode[2] = cudaAddressModeBorder;
	gT_coneVolumeTexture.filterMode = cudaFilterModeLinear;
	gT_coneVolumeTexture.normalized = false;

	cudaBindTextureToArray(gT_coneVolumeTexture, array, channelDesc);

	// TODO: error value?

	return true;
}

	// threadIdx: x = ??? detector  (u?)
	//            y = relative angle

	// blockIdx:  x = ??? detector  (u+v?)
    //            y = angle block


#define CONE_FP_BODY(c0,c1,c2) \
	int angle = startAngle + blockIdx.y * g_anglesPerBlock + threadIdx.y;                                          \
	if (angle >= endAngle)                                                                                         \
		return;                                                                                                    \
                                                                                                                   \
	const float fSrcX = gC_SrcX[angle];                                                                            \
	const float fSrcY = gC_SrcY[angle];                                                                            \
	const float fSrcZ = gC_SrcZ[angle];                                                                            \
	const float fDetUX = gC_DetUX[angle];                                                                          \
	const float fDetUY = gC_DetUY[angle];                                                                          \
	const float fDetUZ = gC_DetUZ[angle];                                                                          \
	const float fDetVX = gC_DetVX[angle];                                                                          \
	const float fDetVY = gC_DetVY[angle];                                                                          \
	const float fDetVZ = gC_DetVZ[angle];                                                                          \
	const float fDetSX = gC_DetSX[angle] + 0.5f * fDetUX + 0.5f * fDetVX;                                          \
	const float fDetSY = gC_DetSY[angle] + 0.5f * fDetUY + 0.5f * fDetVY;                                          \
	const float fDetSZ = gC_DetSZ[angle] + 0.5f * fDetUZ + 0.5f * fDetVZ;                                          \
                                                                                                                   \
	const int detectorU = (blockIdx.x%((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockU + threadIdx.x;      \
	const int startDetectorV = (blockIdx.x/((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockV;               \
	int endDetectorV = startDetectorV + g_detBlockV;                                                               \
	if (endDetectorV > dims.iProjV)                                                                                \
		endDetectorV = dims.iProjV;                                                                                \
                                                                                                                   \
	int endSlice = startSlice + g_blockSlices;                                                                     \
	if (endSlice > dims.iVol##c0)                                                                                     \
		endSlice = dims.iVol##c0;                                                                                    \
                                                                                                                   \
	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)                                    \
	{                                                                                                              \
		/* Trace ray from Src to (detectorU,detectorV) from */                                                     \
		/* X = startSlice to X = endSlice                   */                                                     \
                                                                                                                   \
		const float fDetX = fDetSX + detectorU*fDetUX + detectorV*fDetVX;                                          \
		const float fDetY = fDetSY + detectorU*fDetUY + detectorV*fDetVY;                                          \
		const float fDetZ = fDetSZ + detectorU*fDetUZ + detectorV*fDetVZ;                                          \
                                                                                                                   \
		/*        (x)   ( 1)       ( 0) */                                                                         \
		/* ray:   (y) = (ay) * x + (by) */                                                                         \
		/*        (z)   (az)       (bz) */                                                                         \
                                                                                                                   \
		const float a##c1 = (fSrc##c1 - fDet##c1) / (fSrc##c0 - fDet##c0);                                         \
		const float a##c2 = (fSrc##c2 - fDet##c2) / (fSrc##c0 - fDet##c0);                                         \
		const float b##c1 = fSrc##c1 - a##c1 * fSrc##c0;                                                           \
		const float b##c2 = fSrc##c2 - a##c2 * fSrc##c0;                                                           \
                                                                                                                   \
		const float fDistCorr = sqrt(a##c1*a##c1+a##c2*a##c2+1.0f) * fOutputScale;                                 \
                                                                                                                   \
		float fVal = 0.0f;                                                                                         \
                                                                                                                   \
		float f##c0 = startSlice + 0.5f;                                                                           \
		float f##c1 = a##c1 * (startSlice - 0.5f*dims.iVol##c0 + 0.5f) + b##c1 + 0.5f*dims.iVol##c1 - 0.5f + 0.5f; \
		float f##c2 = a##c2 * (startSlice - 0.5f*dims.iVol##c0 + 0.5f) + b##c2 + 0.5f*dims.iVol##c2 - 0.5f + 0.5f; \
                                                                                                                   \
		for (int s = startSlice; s < endSlice; ++s)                                                                \
		{                                                                                                          \
			fVal += tex3D(gT_coneVolumeTexture, fX, fY, fZ);                                                       \
			f##c0 += 1.0f;                                                                                         \
			f##c1 += a##c1;                                                                                        \
			f##c2 += a##c2;                                                                                        \
		}                                                                                                          \
                                                                                                                   \
		fVal *= fDistCorr;                                                                                         \
                                                                                                                   \
		D_projData[(detectorV*dims.iProjAngles+angle)*projPitch+detectorU] += fVal;                                \
	}

#define CONE_FP_SS_BODY(c0,c1,c2) \
	int angle = startAngle + blockIdx.y * g_anglesPerBlock + threadIdx.y;                                          \
	if (angle >= endAngle)                                                                                         \
		return;                                                                                                    \
                                                                                                                   \
	const float fSrcX = gC_SrcX[angle];                                                                            \
	const float fSrcY = gC_SrcY[angle];                                                                            \
	const float fSrcZ = gC_SrcZ[angle];                                                                            \
	const float fDetUX = gC_DetUX[angle];                                                                          \
	const float fDetUY = gC_DetUY[angle];                                                                          \
	const float fDetUZ = gC_DetUZ[angle];                                                                          \
	const float fDetVX = gC_DetVX[angle];                                                                          \
	const float fDetVY = gC_DetVY[angle];                                                                          \
	const float fDetVZ = gC_DetVZ[angle];                                                                          \
	const float fDetSX = gC_DetSX[angle] + 0.5f * fDetUX + 0.5f * fDetVX;                                          \
	const float fDetSY = gC_DetSY[angle] + 0.5f * fDetUY + 0.5f * fDetVY;                                          \
	const float fDetSZ = gC_DetSZ[angle] + 0.5f * fDetUZ + 0.5f * fDetVZ;                                          \
                                                                                                                   \
	const int detectorU = (blockIdx.x%((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockU + threadIdx.x;      \
	const int startDetectorV = (blockIdx.x/((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockV;               \
	int endDetectorV = startDetectorV + g_detBlockV;                                                               \
	if (endDetectorV > dims.iProjV)                                                                                \
		endDetectorV = dims.iProjV;                                                                                \
                                                                                                                   \
	int endSlice = startSlice + g_blockSlices;                                                                     \
	if (endSlice > dims.iVolX)                                                                                     \
		endSlice = dims.iVolX;                                                                                     \
                                                                                                                   \
	const float fSubStep = 1.0f/dims.iRaysPerDetDim;                                                               \
                                                                                                                   \
	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)                                    \
	{                                                                                                              \
		/* Trace ray from Src to (detectorU,detectorV) from */                                                     \
		/* X = startSlice to X = endSlice                   */                                                     \
                                                                                                                   \
		float fV = 0.0f;                                                                                           \
                                                                                                                   \
		float fdU = detectorU - 0.5f + 0.5f*fSubStep;                                                              \
		for (int iSubU = 0; iSubU < dims.iRaysPerDetDim; ++iSubU, fdU+=fSubStep) {                                 \
		float fdV = detectorV - 0.5f + 0.5f*fSubStep;                                                              \
		for (int iSubV = 0; iSubV < dims.iRaysPerDetDim; ++iSubV, fdV+=fSubStep) {                                 \
                                                                                                                   \
		const float fDetX = fDetSX + fdU*fDetUX + fdV*fDetVX;                                                      \
		const float fDetY = fDetSY + fdU*fDetUY + fdV*fDetVY;                                                      \
		const float fDetZ = fDetSZ + fdU*fDetUZ + fdV*fDetVZ;                                                      \
                                                                                                                   \
		/*        (x)   ( 1)       ( 0) */                                                                         \
		/* ray:   (y) = (ay) * x + (by) */                                                                         \
		/*        (z)   (az)       (bz) */                                                                         \
                                                                                                                   \
		const float a##c1 = (fSrc##c1 - fDet##c1) / (fSrc##c0 - fDet##c0);                                         \
		const float a##c2 = (fSrc##c2 - fDet##c2) / (fSrc##c0 - fDet##c0);                                         \
		const float b##c1 = fSrc##c1 - a##c1 * fSrc##c0;                                                           \
		const float b##c2 = fSrc##c2 - a##c2 * fSrc##c0;                                                           \
                                                                                                                   \
		const float fDistCorr = sqrt(a##c1*a##c1+a##c2*a##c2+1.0f) * fOutputScale;                                 \
                                                                                                                   \
		float fVal = 0.0f;                                                                                         \
                                                                                                                   \
		float f##c0 = startSlice + 0.5f;                                                                           \
		float f##c1 = a##c1 * (startSlice - 0.5f*dims.iVol##c0 + 0.5f) + b##c1 + 0.5f*dims.iVol##c1 - 0.5f + 0.5f; \
		float f##c2 = a##c2 * (startSlice - 0.5f*dims.iVol##c0 + 0.5f) + b##c2 + 0.5f*dims.iVol##c2 - 0.5f + 0.5f; \
                                                                                                                   \
		for (int s = startSlice; s < endSlice; ++s)                                                                \
		{                                                                                                          \
			fVal += tex3D(gT_coneVolumeTexture, fX, fY, fZ);                                                       \
			f##c0 += 1.0f;                                                                                         \
			f##c1 += a##c1;                                                                                        \
			f##c2 += a##c2;                                                                                        \
		}                                                                                                          \
                                                                                                                   \
		fVal *= fDistCorr;                                                                                         \
		fV += fVal;                                                                                                \
                                                                                                                   \
		}                                                                                                          \
		}                                                                                                          \
                                                                                                                   \
		D_projData[(detectorV*dims.iProjAngles+angle)*projPitch+detectorU] += fV / (dims.iRaysPerDetDim * dims.iRaysPerDetDim);\
	}




 
__global__ void FP_dirX(float* D_projData, unsigned int projPitch, unsigned int startSlice, unsigned int startAngle, unsigned int endAngle, const SDimensions3D dims, float fOutputScale)
{
CONE_FP_BODY(X,Y,Z)
}

__global__ void FP_dirY(float* D_projData, unsigned int projPitch, unsigned int startSlice, unsigned int startAngle, unsigned int endAngle, const SDimensions3D dims, float fOutputScale)
{
CONE_FP_BODY(Y,X,Z)
}

__global__ void FP_dirZ(float* D_projData, unsigned int projPitch, unsigned int startSlice, unsigned int startAngle, unsigned int endAngle, const SDimensions3D dims, float fOutputScale)
{
CONE_FP_BODY(Z,X,Y)
}

 
__global__ void FP_SS_dirX(float* D_projData, unsigned int projPitch, unsigned int startSlice, unsigned int startAngle, unsigned int endAngle, const SDimensions3D dims, float fOutputScale)
{
CONE_FP_SS_BODY(X,Y,Z)
}

__global__ void FP_SS_dirY(float* D_projData, unsigned int projPitch, unsigned int startSlice, unsigned int startAngle, unsigned int endAngle, const SDimensions3D dims, float fOutputScale)
{
CONE_FP_SS_BODY(Y,X,Z)
}

__global__ void FP_SS_dirZ(float* D_projData, unsigned int projPitch, unsigned int startSlice, unsigned int startAngle, unsigned int endAngle, const SDimensions3D dims, float fOutputScale)
{
CONE_FP_SS_BODY(Z,X,Y)
}



bool ConeFP_Array(cudaArray *D_volArray,
                  cudaPitchedPtr D_projData,
                  const SDimensions3D& dims, const SConeProjection* angles,
                  float fOutputScale)
{
	bindVolumeDataTexture(D_volArray);

	// transfer angles to constant memory
	float* tmp = new float[dims.iProjAngles];

#define TRANSFER_TO_CONSTANT(name) do { for (unsigned int i = 0; i < dims.iProjAngles; ++i) tmp[i] = angles[i].f##name ; cudaMemcpyToSymbol(gC_##name, tmp, dims.iProjAngles*sizeof(float), 0, cudaMemcpyHostToDevice); } while (0)

	TRANSFER_TO_CONSTANT(SrcX);
	TRANSFER_TO_CONSTANT(SrcY);
	TRANSFER_TO_CONSTANT(SrcZ);
	TRANSFER_TO_CONSTANT(DetSX);
	TRANSFER_TO_CONSTANT(DetSY);
	TRANSFER_TO_CONSTANT(DetSZ);
	TRANSFER_TO_CONSTANT(DetUX);
	TRANSFER_TO_CONSTANT(DetUY);
	TRANSFER_TO_CONSTANT(DetUZ);
	TRANSFER_TO_CONSTANT(DetVX);
	TRANSFER_TO_CONSTANT(DetVY);
	TRANSFER_TO_CONSTANT(DetVZ);

#undef TRANSFER_TO_CONSTANT

	delete[] tmp;

	std::list<cudaStream_t> streams;
	dim3 dimBlock(g_detBlockU, g_anglesPerBlock); // region size, angles

	// Run over all angles, grouping them into groups of the same
	// orientation (roughly horizontal vs. roughly vertical).
	// Start a stream of grids for each such group.

	unsigned int blockStart = 0;
	unsigned int blockEnd = 0;
	int blockDirection = 0;

	// timeval t;
	// tic(t);

	for (unsigned int a = 0; a <= dims.iProjAngles; ++a) {
		int dir;
		if (a != dims.iProjAngles) {
			float dX = fabsf(angles[a].fSrcX - (angles[a].fDetSX + dims.iProjU*angles[a].fDetUX*0.5f + dims.iProjV*angles[a].fDetVX*0.5f));
			float dY = fabsf(angles[a].fSrcY - (angles[a].fDetSY + dims.iProjU*angles[a].fDetUY*0.5f + dims.iProjV*angles[a].fDetVY*0.5f));
			float dZ = fabsf(angles[a].fSrcZ - (angles[a].fDetSZ + dims.iProjU*angles[a].fDetUZ*0.5f + dims.iProjV*angles[a].fDetVZ*0.5f));

			if (dX >= dY && dX >= dZ)
				dir = 0;
			else if (dY >= dX && dY >= dZ)
				dir = 1;
			else
				dir = 2;
		}

		if (a == dims.iProjAngles || dir != blockDirection) {
			// block done

			blockEnd = a;
			if (blockStart != blockEnd) {

				dim3 dimGrid(
				             ((dims.iProjU+g_detBlockU-1)/g_detBlockU)*((dims.iProjV+g_detBlockV-1)/g_detBlockV),
(blockEnd-blockStart+g_anglesPerBlock-1)/g_anglesPerBlock);
				// TODO: check if we can't immediately
				//       destroy the stream after use
				cudaStream_t stream;
				cudaStreamCreate(&stream);
				streams.push_back(stream);

				// printf("angle block: %d to %d, %d (%dx%d, %dx%d)\n", blockStart, blockEnd, blockDirection, dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

				if (blockDirection == 0) {
					for (unsigned int i = 0; i < dims.iVolX; i += g_blockSlices)
						if (dims.iRaysPerDetDim == 1)
							FP_dirX<<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, fOutputScale);
						else
							FP_SS_dirX<<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, fOutputScale);
				} else if (blockDirection == 1) {
					for (unsigned int i = 0; i < dims.iVolY; i += g_blockSlices)
						if (dims.iRaysPerDetDim == 1)
							FP_dirY<<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, fOutputScale);
						else
							FP_SS_dirY<<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, fOutputScale);
				} else if (blockDirection == 2) {
					for (unsigned int i = 0; i < dims.iVolZ; i += g_blockSlices)
						if (dims.iRaysPerDetDim == 1)
							FP_dirZ<<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, fOutputScale);
						else
							FP_SS_dirZ<<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, fOutputScale);
				}

			}

			blockDirection = dir;
			blockStart = a;
		}
	}

	for (std::list<cudaStream_t>::iterator iter = streams.begin(); iter != streams.end(); ++iter)
		cudaStreamDestroy(*iter);

	streams.clear();

	cudaTextForceKernelsCompletion();

	// printf("%f\n", toc(t));

	return true;
}

bool ConeFP(cudaPitchedPtr D_volumeData,
            cudaPitchedPtr D_projData,
            const SDimensions3D& dims, const SConeProjection* angles,
            float fOutputScale)
{
	// transfer volume to array

	cudaArray* cuArray = allocateVolumeArray(dims);
	transferVolumeToArray(D_volumeData, cuArray, dims);

	bool ret = ConeFP_Array(cuArray, D_projData, dims, angles, fOutputScale);

	cudaFreeArray(cuArray);

	return ret;
}


}

#ifdef STANDALONE
int main()
{
	SDimensions3D dims;
	dims.iVolX = 256;
	dims.iVolY = 256;
	dims.iVolZ = 256;
	dims.iProjAngles = 32;
	dims.iProjU = 512;
	dims.iProjV = 512;
	dims.iRaysPerDet = 1;

	cudaExtent extentV;
	extentV.width = dims.iVolX*sizeof(float);
	extentV.height = dims.iVolY;
	extentV.depth = dims.iVolZ;

	cudaPitchedPtr volData; // pitch, ptr, xsize, ysize

	cudaMalloc3D(&volData, extentV);

	cudaExtent extentP;
	extentP.width = dims.iProjU*sizeof(float);
	extentP.height = dims.iProjV;
	extentP.depth = dims.iProjAngles;

	cudaPitchedPtr projData; // pitch, ptr, xsize, ysize

	cudaMalloc3D(&projData, extentP);
	cudaMemset3D(projData, 0, extentP);

	float* slice = new float[256*256];
	cudaPitchedPtr ptr;
	ptr.ptr = slice;
	ptr.pitch = 256*sizeof(float);
	ptr.xsize = 256*sizeof(float);
	ptr.ysize = 256;

	for (unsigned int i = 0; i < 256*256; ++i)
		slice[i] = 1.0f;
	for (unsigned int i = 0; i < 256; ++i) {
		cudaExtent extentS;
		extentS.width = dims.iVolX*sizeof(float);
		extentS.height = dims.iVolY;
		extentS.depth = 1;
		cudaPos sp = { 0, 0, 0 };
		cudaPos dp = { 0, 0, i };
		cudaMemcpy3DParms p;
		p.srcArray = 0;
		p.srcPos = sp;
		p.srcPtr = ptr;
		p.dstArray = 0;
		p.dstPos = dp;
		p.dstPtr = volData;
		p.extent = extentS;
		p.kind = cudaMemcpyHostToDevice;
		cudaError err = cudaMemcpy3D(&p);
		assert(!err);
	}


	SConeProjection angle[32];
	angle[0].fSrcX = -1536;
	angle[0].fSrcY = 0;
	angle[0].fSrcZ = 200;

	angle[0].fDetSX = 512;
	angle[0].fDetSY = -256;
	angle[0].fDetSZ = -256;

	angle[0].fDetUX = 0;
	angle[0].fDetUY = 1;
	angle[0].fDetUZ = 0;

	angle[0].fDetVX = 0;
	angle[0].fDetVY = 0;
	angle[0].fDetVZ = 1;

#define ROTATE0(name,i,alpha) do { angle[i].f##name##X = angle[0].f##name##X * cos(alpha) - angle[0].f##name##Y * sin(alpha); angle[i].f##name##Y = angle[0].f##name##X * sin(alpha) + angle[0].f##name##Y * cos(alpha); } while(0)
	for (int i = 1; i < 32; ++i) {
		angle[i] = angle[0];
		ROTATE0(Src, i, i*1*M_PI/180);
		ROTATE0(DetS, i, i*1*M_PI/180);
		ROTATE0(DetU, i, i*1*M_PI/180);
		ROTATE0(DetV, i, i*1*M_PI/180);
	}
#undef ROTATE0

	astraCUDA3d::ConeFP(volData, projData, dims, angle, 1.0f);

	float* buf = new float[512*512];

	cudaMemcpy(buf, ((float*)projData.ptr)+512*512*8, 512*512*sizeof(float), cudaMemcpyDeviceToHost);

	printf("%d %d %d\n", projData.pitch, projData.xsize, projData.ysize);

	saveImage("proj.png", 512, 512, buf);
	

}
#endif
