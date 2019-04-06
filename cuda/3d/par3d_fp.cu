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

#include "astra/cuda/3d/util3d.h"
#include "astra/cuda/3d/dims3d.h"

#include <cstdio>
#include <cassert>
#include <iostream>
#include <list>

#include <cuda.h>

typedef texture<float, 3, cudaReadModeElementType> texture3D;

static texture3D gT_par3DVolumeTexture;

namespace astraCUDA3d {

static const unsigned int g_anglesPerBlock = 4;

// thickness of the slices we're splitting the volume up into
static const unsigned int g_blockSlices = 32;
static const unsigned int g_detBlockU = 32;
static const unsigned int g_detBlockV = 32;

static const unsigned g_MaxAngles = 1024;
__constant__ float gC_RayX[g_MaxAngles];
__constant__ float gC_RayY[g_MaxAngles];
__constant__ float gC_RayZ[g_MaxAngles];
__constant__ float gC_DetSX[g_MaxAngles];
__constant__ float gC_DetSY[g_MaxAngles];
__constant__ float gC_DetSZ[g_MaxAngles];
__constant__ float gC_DetUX[g_MaxAngles];
__constant__ float gC_DetUY[g_MaxAngles];
__constant__ float gC_DetUZ[g_MaxAngles];
__constant__ float gC_DetVX[g_MaxAngles];
__constant__ float gC_DetVY[g_MaxAngles];
__constant__ float gC_DetVZ[g_MaxAngles];


static bool bindVolumeDataTexture(const cudaArray* array)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	gT_par3DVolumeTexture.addressMode[0] = cudaAddressModeBorder;
	gT_par3DVolumeTexture.addressMode[1] = cudaAddressModeBorder;
	gT_par3DVolumeTexture.addressMode[2] = cudaAddressModeBorder;
	gT_par3DVolumeTexture.filterMode = cudaFilterModeLinear;
	gT_par3DVolumeTexture.normalized = false;

	cudaBindTextureToArray(gT_par3DVolumeTexture, array, channelDesc);

	// TODO: error value?

	return true;
}


// x=0, y=1, z=2
struct DIR_X {
	__device__ float nSlices(const SDimensions3D& dims) const { return dims.iVolX; }
	__device__ float nDim1(const SDimensions3D& dims) const { return dims.iVolY; }
	__device__ float nDim2(const SDimensions3D& dims) const { return dims.iVolZ; }
	__device__ float c0(float x, float y, float z) const { return x; }
	__device__ float c1(float x, float y, float z) const { return y; }
	__device__ float c2(float x, float y, float z) const { return z; }
	__device__ float tex(float f0, float f1, float f2) const { return tex3D(gT_par3DVolumeTexture, f0, f1, f2); }
	__device__ float x(float f0, float f1, float f2) const { return f0; }
	__device__ float y(float f0, float f1, float f2) const { return f1; }
	__device__ float z(float f0, float f1, float f2) const { return f2; }
};

// y=0, x=1, z=2
struct DIR_Y {
	__device__ float nSlices(const SDimensions3D& dims) const { return dims.iVolY; }
	__device__ float nDim1(const SDimensions3D& dims) const { return dims.iVolX; }
	__device__ float nDim2(const SDimensions3D& dims) const { return dims.iVolZ; }
	__device__ float c0(float x, float y, float z) const { return y; }
	__device__ float c1(float x, float y, float z) const { return x; }
	__device__ float c2(float x, float y, float z) const { return z; }
	__device__ float tex(float f0, float f1, float f2) const { return tex3D(gT_par3DVolumeTexture, f1, f0, f2); }
	__device__ float x(float f0, float f1, float f2) const { return f1; }
	__device__ float y(float f0, float f1, float f2) const { return f0; }
	__device__ float z(float f0, float f1, float f2) const { return f2; }
};

// z=0, x=1, y=2
struct DIR_Z {
	__device__ float nSlices(const SDimensions3D& dims) const { return dims.iVolZ; }
	__device__ float nDim1(const SDimensions3D& dims) const { return dims.iVolX; }
	__device__ float nDim2(const SDimensions3D& dims) const { return dims.iVolY; }
	__device__ float c0(float x, float y, float z) const { return z; }
	__device__ float c1(float x, float y, float z) const { return x; }
	__device__ float c2(float x, float y, float z) const { return y; }
	__device__ float tex(float f0, float f1, float f2) const { return tex3D(gT_par3DVolumeTexture, f1, f2, f0); }
	__device__ float x(float f0, float f1, float f2) const { return f1; }
	__device__ float y(float f0, float f1, float f2) const { return f2; }
	__device__ float z(float f0, float f1, float f2) const { return f0; }
};

struct SCALE_CUBE {
	float fOutputScale;
	__device__ float scale(float a1, float a2) const { return sqrt(a1*a1+a2*a2+1.0f) * fOutputScale; }
};

struct SCALE_NONCUBE {
	float fScale1;
	float fScale2;
	float fOutputScale;
	__device__ float scale(float a1, float a2) const { return sqrt(a1*a1*fScale1+a2*a2*fScale2+1.0f) * fOutputScale; }
};



// threadIdx: x = u detector
//            y = relative angle
// blockIdx:  x = u/v detector
//            y = angle block


template<class COORD, class SCALE>
__global__ void par3D_FP_t(float* D_projData, unsigned int projPitch,
                           unsigned int startSlice,
                           unsigned int startAngle, unsigned int endAngle,
                           const SDimensions3D dims,
                           SCALE sc)
{
	COORD c;

	int angle = startAngle + blockIdx.y * g_anglesPerBlock + threadIdx.y;
	if (angle >= endAngle)
		return;

	const float fRayX = gC_RayX[angle];
	const float fRayY = gC_RayY[angle];
	const float fRayZ = gC_RayZ[angle];
	const float fDetUX = gC_DetUX[angle];
	const float fDetUY = gC_DetUY[angle];
	const float fDetUZ = gC_DetUZ[angle];
	const float fDetVX = gC_DetVX[angle];
	const float fDetVY = gC_DetVY[angle];
	const float fDetVZ = gC_DetVZ[angle];
	const float fDetSX = gC_DetSX[angle] + 0.5f * fDetUX + 0.5f * fDetVX;
	const float fDetSY = gC_DetSY[angle] + 0.5f * fDetUY + 0.5f * fDetVY;
	const float fDetSZ = gC_DetSZ[angle] + 0.5f * fDetUZ + 0.5f * fDetVZ;

	const float a1 = c.c1(fRayX,fRayY,fRayZ) / c.c0(fRayX,fRayY,fRayZ);
	const float a2 = c.c2(fRayX,fRayY,fRayZ) / c.c0(fRayX,fRayY,fRayZ);
	const float fDistCorr = sc.scale(a1, a2);


	const int detectorU = (blockIdx.x%((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockU + threadIdx.x;
	const int startDetectorV = (blockIdx.x/((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockV;
	int endDetectorV = startDetectorV + g_detBlockV;
	if (endDetectorV > dims.iProjV)
		endDetectorV = dims.iProjV;

	int endSlice = startSlice + g_blockSlices;
	if (endSlice > c.nSlices(dims))
		endSlice = c.nSlices(dims);

	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)
	{
		/* Trace ray in direction Ray to (detectorU,detectorV) from  */
		/* X = startSlice to X = endSlice                            */

		const float fDetX = fDetSX + detectorU*fDetUX + detectorV*fDetVX;
		const float fDetY = fDetSY + detectorU*fDetUY + detectorV*fDetVY;
		const float fDetZ = fDetSZ + detectorU*fDetUZ + detectorV*fDetVZ;

		/*        (x)   ( 1)       ( 0)    */
		/* ray:   (y) = (ay) * x + (by)    */
		/*        (z)   (az)       (bz)    */

		const float b1 = c.c1(fDetX,fDetY,fDetZ) - a1 * c.c0(fDetX,fDetY,fDetZ);
		const float b2 = c.c2(fDetX,fDetY,fDetZ) - a2 * c.c0(fDetX,fDetY,fDetZ);

		float fVal = 0.0f;

		float f0 = startSlice + 0.5f;
		float f1 = a1 * (startSlice - 0.5f*c.nSlices(dims) + 0.5f) + b1 + 0.5f*c.nDim1(dims) - 0.5f + 0.5f;
		float f2 = a2 * (startSlice - 0.5f*c.nSlices(dims) + 0.5f) + b2 + 0.5f*c.nDim2(dims) - 0.5f + 0.5f;

		for (int s = startSlice; s < endSlice; ++s)
		{
			fVal += c.tex(f0, f1, f2);
			f0 += 1.0f;
			f1 += a1;
			f2 += a2;
		}

		fVal *= fDistCorr;

		D_projData[(detectorV*dims.iProjAngles+angle)*projPitch+detectorU] += fVal;
	}
}

// Supersampling version
template<class COORD>
__global__ void par3D_FP_SS_t(float* D_projData, unsigned int projPitch,
                              unsigned int startSlice,
                              unsigned int startAngle, unsigned int endAngle,
                              const SDimensions3D dims, int iRaysPerDetDim,
                              SCALE_NONCUBE sc)
{
	COORD c;

	int angle = startAngle + blockIdx.y * g_anglesPerBlock + threadIdx.y;
	if (angle >= endAngle)
		return;

	const float fRayX = gC_RayX[angle];
	const float fRayY = gC_RayY[angle];
	const float fRayZ = gC_RayZ[angle];
	const float fDetUX = gC_DetUX[angle];
	const float fDetUY = gC_DetUY[angle];
	const float fDetUZ = gC_DetUZ[angle];
	const float fDetVX = gC_DetVX[angle];
	const float fDetVY = gC_DetVY[angle];
	const float fDetVZ = gC_DetVZ[angle];
	const float fDetSX = gC_DetSX[angle] + 0.5f * fDetUX + 0.5f * fDetVX;
	const float fDetSY = gC_DetSY[angle] + 0.5f * fDetUY + 0.5f * fDetVY;
	const float fDetSZ = gC_DetSZ[angle] + 0.5f * fDetUZ + 0.5f * fDetVZ;

	const float a1 = c.c1(fRayX,fRayY,fRayZ) / c.c0(fRayX,fRayY,fRayZ);
	const float a2 = c.c2(fRayX,fRayY,fRayZ) / c.c0(fRayX,fRayY,fRayZ);
	const float fDistCorr = sc.scale(a1, a2);

	const int detectorU = (blockIdx.x%((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockU + threadIdx.x;
	const int startDetectorV = (blockIdx.x/((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockV;
	int endDetectorV = startDetectorV + g_detBlockV;
	if (endDetectorV > dims.iProjV)
		endDetectorV = dims.iProjV;

	int endSlice = startSlice + g_blockSlices;
	if (endSlice > c.nSlices(dims))
		endSlice = c.nSlices(dims);

	const float fSubStep = 1.0f/iRaysPerDetDim;

	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)
	{

		float fV = 0.0f;

		float fdU = detectorU - 0.5f + 0.5f*fSubStep;
		for (int iSubU = 0; iSubU < iRaysPerDetDim; ++iSubU, fdU+=fSubStep) {
		float fdV = detectorV - 0.5f + 0.5f*fSubStep;
		for (int iSubV = 0; iSubV < iRaysPerDetDim; ++iSubV, fdV+=fSubStep) {

		/* Trace ray in direction Ray to (detectorU,detectorV) from  */
		/* X = startSlice to X = endSlice                            */

		const float fDetX = fDetSX + fdU*fDetUX + fdV*fDetVX;
		const float fDetY = fDetSY + fdU*fDetUY + fdV*fDetVY;
		const float fDetZ = fDetSZ + fdU*fDetUZ + fdV*fDetVZ;

		/*        (x)   ( 1)       ( 0)    */
		/* ray:   (y) = (ay) * x + (by)    */
		/*        (z)   (az)       (bz)    */

		const float b1 = c.c1(fDetX,fDetY,fDetZ) - a1 * c.c0(fDetX,fDetY,fDetZ);
		const float b2 = c.c2(fDetX,fDetY,fDetZ) - a2 * c.c0(fDetX,fDetY,fDetZ);


		float fVal = 0.0f;

		float f0 = startSlice + 0.5f;
		float f1 = a1 * (startSlice - 0.5f*c.nSlices(dims) + 0.5f) + b1 + 0.5f*c.nDim1(dims) - 0.5f + 0.5f;
		float f2 = a2 * (startSlice - 0.5f*c.nSlices(dims) + 0.5f) + b2 + 0.5f*c.nDim2(dims) - 0.5f + 0.5f;

		for (int s = startSlice; s < endSlice; ++s)
		{
			fVal += c.tex(f0, f1, f2);
			f0 += 1.0f;
			f1 += a1;
			f2 += a2;
		}

		fV += fVal;

		}
		}

		fV *= fDistCorr;
		D_projData[(detectorV*dims.iProjAngles+angle)*projPitch+detectorU] += fV / (iRaysPerDetDim * iRaysPerDetDim);
	}
}


__device__ float dirWeights(float fX, float fN) {
	if (fX <= -0.5f) // outside image on left
		return 0.0f;
	if (fX <= 0.5f) // half outside image on left
		return (fX + 0.5f) * (fX + 0.5f);
	if (fX <= fN - 0.5f) { // inside image
		float t = fX + 0.5f - floorf(fX + 0.5f);
		return t*t + (1-t)*(1-t);
	}
	if (fX <= fN + 0.5f) // half outside image on right
		return (fN + 0.5f - fX) * (fN + 0.5f - fX);
	return 0.0f; // outside image on right
}

template<class COORD>
__global__ void par3D_FP_SumSqW_t(float* D_projData, unsigned int projPitch,
                                  unsigned int startSlice,
                                  unsigned int startAngle, unsigned int endAngle,
                                  const SDimensions3D dims,
                                  SCALE_NONCUBE sc)
{
	COORD c;

	int angle = startAngle + blockIdx.y * g_anglesPerBlock + threadIdx.y;
	if (angle >= endAngle)
		return;

	const float fRayX = gC_RayX[angle];
	const float fRayY = gC_RayY[angle];
	const float fRayZ = gC_RayZ[angle];
	const float fDetUX = gC_DetUX[angle];
	const float fDetUY = gC_DetUY[angle];
	const float fDetUZ = gC_DetUZ[angle];
	const float fDetVX = gC_DetVX[angle];
	const float fDetVY = gC_DetVY[angle];
	const float fDetVZ = gC_DetVZ[angle];
	const float fDetSX = gC_DetSX[angle] + 0.5f * fDetUX + 0.5f * fDetVX;
	const float fDetSY = gC_DetSY[angle] + 0.5f * fDetUY + 0.5f * fDetVY;
	const float fDetSZ = gC_DetSZ[angle] + 0.5f * fDetUZ + 0.5f * fDetVZ;

	const float a1 = c.c1(fRayX,fRayY,fRayZ) / c.c0(fRayX,fRayY,fRayZ);
	const float a2 = c.c2(fRayX,fRayY,fRayZ) / c.c0(fRayX,fRayY,fRayZ);
	const float fDistCorr = sc.scale(a1, a2);


	const int detectorU = (blockIdx.x%((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockU + threadIdx.x;
	const int startDetectorV = (blockIdx.x/((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockV;
	int endDetectorV = startDetectorV + g_detBlockV;
	if (endDetectorV > dims.iProjV)
		endDetectorV = dims.iProjV;

	int endSlice = startSlice + g_blockSlices;
	if (endSlice > c.nSlices(dims))
		endSlice = c.nSlices(dims);

	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)
	{
		/* Trace ray in direction Ray to (detectorU,detectorV) from  */
		/* X = startSlice to X = endSlice                            */

		const float fDetX = fDetSX + detectorU*fDetUX + detectorV*fDetVX;
		const float fDetY = fDetSY + detectorU*fDetUY + detectorV*fDetVY;
		const float fDetZ = fDetSZ + detectorU*fDetUZ + detectorV*fDetVZ;

		/*        (x)   ( 1)       ( 0)    */
		/* ray:   (y) = (ay) * x + (by)    */
		/*        (z)   (az)       (bz)    */

		const float b1 = c.c1(fDetX,fDetY,fDetZ) - a1 * c.c0(fDetX,fDetY,fDetZ);
		const float b2 = c.c2(fDetX,fDetY,fDetZ) - a2 * c.c0(fDetX,fDetY,fDetZ);

		float fVal = 0.0f;

		float f0 = startSlice + 0.5f;
		float f1 = a1 * (startSlice - 0.5f*c.nSlices(dims) + 0.5f) + b1 + 0.5f*c.nDim1(dims) - 0.5f + 0.5f;
		float f2 = a2 * (startSlice - 0.5f*c.nSlices(dims) + 0.5f) + b2 + 0.5f*c.nDim2(dims) - 0.5f + 0.5f;

		for (int s = startSlice; s < endSlice; ++s)
		{
			fVal += dirWeights(f1, c.nDim1(dims)) * dirWeights(f2, c.nDim2(dims));
			f0 += 1.0f;
			f1 += a1;
			f2 += a2;
		}

		fVal *= fDistCorr * fDistCorr;
		D_projData[(detectorV*dims.iProjAngles+angle)*projPitch+detectorU] += fVal;
	}
}

// Supersampling version
// TODO


bool Par3DFP_Array_internal(cudaPitchedPtr D_projData,
                   const SDimensions3D& dims, unsigned int angleCount, const SPar3DProjection* angles,
                   const SProjectorParams3D& params)
{
	// transfer angles to constant memory
	float* tmp = new float[dims.iProjAngles];

#define TRANSFER_TO_CONSTANT(name) do { for (unsigned int i = 0; i < angleCount; ++i) tmp[i] = angles[i].f##name ; cudaMemcpyToSymbol(gC_##name, tmp, angleCount*sizeof(float), 0, cudaMemcpyHostToDevice); } while (0)

	TRANSFER_TO_CONSTANT(RayX);
	TRANSFER_TO_CONSTANT(RayY);
	TRANSFER_TO_CONSTANT(RayZ);
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

	bool cube = true;
	if (abs(params.fVolScaleX / params.fVolScaleY - 1.0) > 0.00001)
		cube = false;
	if (abs(params.fVolScaleX / params.fVolScaleZ - 1.0) > 0.00001)
		cube = false;

	SCALE_CUBE scube;
	scube.fOutputScale = params.fOutputScale * params.fVolScaleX;

	SCALE_NONCUBE snoncubeX;
	float fS1 = params.fVolScaleY / params.fVolScaleX;
	snoncubeX.fScale1 = fS1 * fS1;
	float fS2 = params.fVolScaleZ / params.fVolScaleX;
	snoncubeX.fScale2 = fS2 * fS2;
	snoncubeX.fOutputScale = params.fOutputScale * params.fVolScaleX;

	SCALE_NONCUBE snoncubeY;
	fS1 = params.fVolScaleX / params.fVolScaleY;
	snoncubeY.fScale1 = fS1 * fS1;
	fS2 = params.fVolScaleY / params.fVolScaleY;
	snoncubeY.fScale2 = fS2 * fS2;
	snoncubeY.fOutputScale = params.fOutputScale * params.fVolScaleY;

	SCALE_NONCUBE snoncubeZ;
	fS1 = params.fVolScaleX / params.fVolScaleZ;
	snoncubeZ.fScale1 = fS1 * fS1;
	fS2 = params.fVolScaleY / params.fVolScaleZ;
	snoncubeZ.fScale2 = fS2 * fS2;
	snoncubeZ.fOutputScale = params.fOutputScale * params.fVolScaleZ;

	// timeval t;
	// tic(t);

	for (unsigned int a = 0; a <= angleCount; ++a) {
		int dir = -1;
		if (a != dims.iProjAngles) {
			float dX = fabsf(angles[a].fRayX);
			float dY = fabsf(angles[a].fRayY);
			float dZ = fabsf(angles[a].fRayZ);

			if (dX >= dY && dX >= dZ)
				dir = 0;
			else if (dY >= dX && dY >= dZ)
				dir = 1;
			else
				dir = 2;
		}

		if (a == angleCount || dir != blockDirection) {
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
						if (params.iRaysPerDetDim == 1)
								if (cube)
										par3D_FP_t<DIR_X><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, scube);
								else
										par3D_FP_t<DIR_X><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, snoncubeX);
						else
							par3D_FP_SS_t<DIR_X><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, params.iRaysPerDetDim, snoncubeX);
				} else if (blockDirection == 1) {
					for (unsigned int i = 0; i < dims.iVolY; i += g_blockSlices)
						if (params.iRaysPerDetDim == 1)
								if (cube)
										par3D_FP_t<DIR_Y><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, scube);
								else
										par3D_FP_t<DIR_Y><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, snoncubeY);
						else
							par3D_FP_SS_t<DIR_Y><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, params.iRaysPerDetDim, snoncubeY);
				} else if (blockDirection == 2) {
					for (unsigned int i = 0; i < dims.iVolZ; i += g_blockSlices)
						if (params.iRaysPerDetDim == 1)
								if (cube)
										par3D_FP_t<DIR_Z><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, scube);
								else
										par3D_FP_t<DIR_Z><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, snoncubeZ);
						else
							par3D_FP_SS_t<DIR_Z><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, params.iRaysPerDetDim, snoncubeZ);
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

bool Par3DFP(cudaPitchedPtr D_volumeData,
             cudaPitchedPtr D_projData,
             const SDimensions3D& dims, const SPar3DProjection* angles,
             const SProjectorParams3D& params)
{
	// transfer volume to array
	cudaArray* cuArray = allocateVolumeArray(dims);
	transferVolumeToArray(D_volumeData, cuArray, dims);
	bindVolumeDataTexture(cuArray);

	bool ret;

	for (unsigned int iAngle = 0; iAngle < dims.iProjAngles; iAngle += g_MaxAngles) {
		unsigned int iEndAngle = iAngle + g_MaxAngles;
		if (iEndAngle >= dims.iProjAngles)
			iEndAngle = dims.iProjAngles;

		cudaPitchedPtr D_subprojData = D_projData;
		D_subprojData.ptr = (char*)D_projData.ptr + iAngle * D_projData.pitch;

		ret = Par3DFP_Array_internal(D_subprojData,
		                             dims, iEndAngle - iAngle, angles + iAngle,
		                             params);
		if (!ret)
			break;
	}

	cudaFreeArray(cuArray);

	return ret;
}



bool Par3DFP_SumSqW(cudaPitchedPtr D_volumeData,
                    cudaPitchedPtr D_projData,
                    const SDimensions3D& dims, const SPar3DProjection* angles,
                    const SProjectorParams3D& params)
{
	// transfer angles to constant memory
	float* tmp = new float[dims.iProjAngles];

#define TRANSFER_TO_CONSTANT(name) do { for (unsigned int i = 0; i < dims.iProjAngles; ++i) tmp[i] = angles[i].f##name ; cudaMemcpyToSymbol(gC_##name, tmp, dims.iProjAngles*sizeof(float), 0, cudaMemcpyHostToDevice); } while (0)

	TRANSFER_TO_CONSTANT(RayX);
	TRANSFER_TO_CONSTANT(RayY);
	TRANSFER_TO_CONSTANT(RayZ);
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

	SCALE_NONCUBE snoncubeX;
	float fS1 = params.fVolScaleY / params.fVolScaleX;
	snoncubeX.fScale1 = fS1 * fS1;
	float fS2 = params.fVolScaleZ / params.fVolScaleX;
	snoncubeX.fScale2 = fS2 * fS2;
	snoncubeX.fOutputScale = params.fOutputScale * params.fVolScaleX;

	SCALE_NONCUBE snoncubeY;
	fS1 = params.fVolScaleX / params.fVolScaleY;
	snoncubeY.fScale1 = fS1 * fS1;
	fS2 = params.fVolScaleY / params.fVolScaleY;
	snoncubeY.fScale2 = fS2 * fS2;
	snoncubeY.fOutputScale = params.fOutputScale * params.fVolScaleY;

	SCALE_NONCUBE snoncubeZ;
	fS1 = params.fVolScaleX / params.fVolScaleZ;
	snoncubeZ.fScale1 = fS1 * fS1;
	fS2 = params.fVolScaleY / params.fVolScaleZ;
	snoncubeZ.fScale2 = fS2 * fS2;
	snoncubeZ.fOutputScale = params.fOutputScale * params.fVolScaleZ;


	// timeval t;
	// tic(t);

	for (unsigned int a = 0; a <= dims.iProjAngles; ++a) {
		int dir;
		if (a != dims.iProjAngles) {
			float dX = fabsf(angles[a].fRayX);
			float dY = fabsf(angles[a].fRayY);
			float dZ = fabsf(angles[a].fRayZ);

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
						if (params.iRaysPerDetDim == 1)
							par3D_FP_SumSqW_t<DIR_X><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, snoncubeX);
						else
#if 0
							par3D_FP_SS_SumSqW_dirX<<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, fOutputScale);
#else
							assert(false);
#endif
				} else if (blockDirection == 1) {
					for (unsigned int i = 0; i < dims.iVolY; i += g_blockSlices)
						if (params.iRaysPerDetDim == 1)
							par3D_FP_SumSqW_t<DIR_Y><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, snoncubeY);
						else
#if 0
							par3D_FP_SS_SumSqW_dirY<<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, fOutputScale);
#else
							assert(false);
#endif
				} else if (blockDirection == 2) {
					for (unsigned int i = 0; i < dims.iVolZ; i += g_blockSlices)
						if (params.iRaysPerDetDim == 1)
							par3D_FP_SumSqW_t<DIR_Z><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, snoncubeZ);
						else
#if 0
							par3D_FP_SS_SumSqW_dirZ<<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, fOutputScale);
#else
							assert(false);
#endif
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







}
