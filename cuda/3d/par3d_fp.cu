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

#include <cuda.h>
#include "util3d.h"

#ifdef STANDALONE
#include "testutil.h"
#endif

#include "dims3d.h"

#include "astra/MPIProjector3D.h"

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



// threadIdx: x = u detector
//            y = relative angle
// blockIdx:  x = u/v detector
//            y = angle block


template<class COORD>
__global__ void par3D_FP_t(float* D_projData, unsigned int projPitch,
                           unsigned int startSlice,
                           unsigned int startAngle, unsigned int endAngle,
                           const SDimensions3D dims, float fOutputScale)
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

		const float a1 = c.c1(fRayX,fRayY,fRayZ) / c.c0(fRayX,fRayY,fRayZ);
		const float a2 = c.c2(fRayX,fRayY,fRayZ) / c.c0(fRayX,fRayY,fRayZ);
		const float b1 = c.c1(fDetX,fDetY,fDetZ) - a1 * c.c0(fDetX,fDetY,fDetZ);
		const float b2 = c.c2(fDetX,fDetY,fDetZ) - a2 * c.c0(fDetX,fDetY,fDetZ);

		const float fDistCorr = sqrt(a1*a1+a2*a2+1.0f) * fOutputScale;

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
                              const SDimensions3D dims, float fOutputScale)
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



	const int detectorU = (blockIdx.x%((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockU + threadIdx.x;
	const int startDetectorV = (blockIdx.x/((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockV;
	int endDetectorV = startDetectorV + g_detBlockV;
	if (endDetectorV > dims.iProjV)
		endDetectorV = dims.iProjV;

	int endSlice = startSlice + g_blockSlices;
	if (endSlice > c.nSlices(dims))
		endSlice = c.nSlices(dims);

	const float fSubStep = 1.0f/dims.iRaysPerDetDim;

	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)
	{

		float fV = 0.0f;

		float fdU = detectorU - 0.5f + 0.5f*fSubStep;
		for (int iSubU = 0; iSubU < dims.iRaysPerDetDim; ++iSubU, fdU+=fSubStep) {
		float fdV = detectorV - 0.5f + 0.5f*fSubStep;
		for (int iSubV = 0; iSubV < dims.iRaysPerDetDim; ++iSubV, fdV+=fSubStep) {

		/* Trace ray in direction Ray to (detectorU,detectorV) from  */
		/* X = startSlice to X = endSlice                            */

		const float fDetX = fDetSX + fdU*fDetUX + fdV*fDetVX;
		const float fDetY = fDetSY + fdU*fDetUY + fdV*fDetVY;
		const float fDetZ = fDetSZ + fdU*fDetUZ + fdV*fDetVZ;

		/*        (x)   ( 1)       ( 0)    */
		/* ray:   (y) = (ay) * x + (by)    */
		/*        (z)   (az)       (bz)    */

		const float a1 = c.c1(fRayX,fRayY,fRayZ) / c.c0(fRayX,fRayY,fRayZ);
		const float a2 = c.c2(fRayX,fRayY,fRayZ) / c.c0(fRayX,fRayY,fRayZ);
		const float b1 = c.c1(fDetX,fDetY,fDetZ) - a1 * c.c0(fDetX,fDetY,fDetZ);
		const float b2 = c.c2(fDetX,fDetY,fDetZ) - a2 * c.c0(fDetX,fDetY,fDetZ);

		const float fDistCorr = sqrt(a1*a1+a2*a2+1.0f) * fOutputScale;

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
		fV += fVal;

		}
		}

		D_projData[(detectorV*dims.iProjAngles+angle)*projPitch+detectorU] += fV / (dims.iRaysPerDetDim * dims.iRaysPerDetDim);
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
                                  const SDimensions3D dims, float fOutputScale)
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

		const float a1 = c.c1(fRayX,fRayY,fRayZ) / c.c0(fRayX,fRayY,fRayZ);
		const float a2 = c.c2(fRayX,fRayY,fRayZ) / c.c0(fRayX,fRayY,fRayZ);
		const float b1 = c.c1(fDetX,fDetY,fDetZ) - a1 * c.c0(fDetX,fDetY,fDetZ);
		const float b2 = c.c2(fDetX,fDetY,fDetZ) - a2 * c.c0(fDetX,fDetY,fDetZ);

		const float fDistCorr = sqrt(a1*a1+a2*a2+1.0f) * fOutputScale;

		float fVal = 0.0f;

		float f0 = startSlice + 0.5f;
		float f1 = a1 * (startSlice - 0.5f*c.nSlices(dims) + 0.5f) + b1 + 0.5f*c.nDim1(dims) - 0.5f + 0.5f;
		float f2 = a2 * (startSlice - 0.5f*c.nSlices(dims) + 0.5f) + b2 + 0.5f*c.nDim2(dims) - 0.5f + 0.5f;

		for (int s = startSlice; s < endSlice; ++s)
		{
			fVal += dirWeights(f1, c.nDim1(dims)) * dirWeights(f2, c.nDim2(dims)) * fDistCorr * fDistCorr;
			f0 += 1.0f;
			f1 += a1;
			f2 += a2;
		}

		D_projData[(detectorV*dims.iProjAngles+angle)*projPitch+detectorU] += fVal;
	}
}

// Supersampling version
// TODO


bool Par3DFP_Array_internal(cudaPitchedPtr D_projData,
                   const SDimensions3D& dims, unsigned int angleCount, const SPar3DProjection* angles,
                   float fOutputScale)
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
						if (dims.iRaysPerDetDim == 1)
							par3D_FP_t<DIR_X><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, fOutputScale);
						else
							par3D_FP_SS_t<DIR_X><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, fOutputScale);
				} else if (blockDirection == 1) {
					for (unsigned int i = 0; i < dims.iVolY; i += g_blockSlices)
						if (dims.iRaysPerDetDim == 1)
							par3D_FP_t<DIR_Y><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, fOutputScale);
						else
							par3D_FP_SS_t<DIR_Y><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, fOutputScale);
				} else if (blockDirection == 2) {
					for (unsigned int i = 0; i < dims.iVolZ; i += g_blockSlices)
						if (dims.iRaysPerDetDim == 1)
							par3D_FP_t<DIR_Z><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, fOutputScale);
						else
							par3D_FP_SS_t<DIR_Z><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, fOutputScale);
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
             const SDimensions3D& dims2, const SPar3DProjection* angles,
             float fOutputScale,
	     const astra::CMPIProjector3D *mpiPrj = NULL)
{
	int 	       zoffset = 0, incr = 0;
	SDimensions3D  dims    = dims2;

#if USE_MPI
	if(mpiPrj)
	{
		//Modify the volume properties to ignore the ghostcells
		//Modify the height, and change the startpoint of the copy (zoffset)
		int2 ghosts = mpiPrj->getGhostCells();
		dims.iVolZ -= (ghosts.x + ghosts.y);
		zoffset     = ghosts.x;


		//Now for the projection data ghostcells
		ghosts 	     = mpiPrj->getGhostCellsPrj();
		dims.iProjV -= (ghosts.x + ghosts.y);  
		incr         = ghosts.x * D_projData.pitch * D_projData.ysize;


	}
#endif

	// transfer volume to array
	cudaArray* cuArray = allocateVolumeArray(dims);
	transferVolumeToArray(D_volumeData, cuArray, dims, zoffset); //NOTE the zoffset
	bindVolumeDataTexture(cuArray);

	bool ret;

	for (unsigned int iAngle = 0; iAngle < dims.iProjAngles; iAngle += g_MaxAngles) {
		unsigned int iEndAngle = iAngle + g_MaxAngles;
		if (iEndAngle >= dims.iProjAngles)
			iEndAngle = dims.iProjAngles;

		cudaPitchedPtr D_subprojData = D_projData;
		D_subprojData.ptr = (char*)D_projData.ptr + iAngle * D_projData.pitch + incr;

		ret = Par3DFP_Array_internal(D_subprojData,
		                             dims, iEndAngle - iAngle, angles + iAngle,
		                             fOutputScale);
		if (!ret)
			break;
	}

	cudaFreeArray(cuArray);

#if USE_MPI
   	  if(mpiPrj)
	  {
	    //Note not changed ptr as the exchange function figures this out itself
	    const_cast<astra::CMPIProjector3D*>(mpiPrj)->exchangeOverlapRegions(NULL, D_projData, false);
	    const_cast<astra::CMPIProjector3D*>(mpiPrj)->exchangeOverlapAndGhostRegions(NULL, D_projData, false, 1); //1 only the non-overlapped ghost region parts
	  }
#endif
	return ret;
}



bool Par3DFP_SumSqW(cudaPitchedPtr D_volumeData,
                    cudaPitchedPtr D_projData,
                    const SDimensions3D& dims, const SPar3DProjection* angles,
                    float fOutputScale,
	     	    const astra::CMPIProjector3D *mpiPrj = NULL)
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
						if (dims.iRaysPerDetDim == 1)
							par3D_FP_SumSqW_t<DIR_X><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, fOutputScale);
						else
#if 0
							par3D_FP_SS_SumSqW_dirX<<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, fOutputScale);
#else
							assert(false);
#endif
				} else if (blockDirection == 1) {
					for (unsigned int i = 0; i < dims.iVolY; i += g_blockSlices)
						if (dims.iRaysPerDetDim == 1)
							par3D_FP_SumSqW_t<DIR_Y><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, fOutputScale);
						else
#if 0
							par3D_FP_SS_SumSqW_dirY<<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, fOutputScale);
#else
							assert(false);
#endif
				} else if (blockDirection == 2) {
					for (unsigned int i = 0; i < dims.iVolZ; i += g_blockSlices)
						if (dims.iRaysPerDetDim == 1)
							par3D_FP_SumSqW_t<DIR_Z><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), i, blockStart, blockEnd, dims, fOutputScale);
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
#if USE_MPI
   	  if(mpiPrj)
	  {
	    const_cast<astra::CMPIProjector3D*>(mpiPrj)->exchangeOverlapRegions(NULL, D_projData, false);
	    const_cast<astra::CMPIProjector3D*>(mpiPrj)->exchangeOverlapAndGhostRegions(NULL, D_projData, false, 1);
	  }
#endif

	return true;
}







}

#ifdef STANDALONE

using namespace astraCUDA3d;

int main()
{
	cudaSetDevice(1);


	SDimensions3D dims;
	dims.iVolX = 500;
	dims.iVolY = 500;
	dims.iVolZ = 81;
	dims.iProjAngles = 241;
	dims.iProjU = 600;
	dims.iProjV = 100;
	dims.iRaysPerDet = 1;

	SPar3DProjection base;
	base.fRayX = 1.0f;
	base.fRayY = 0.0f;
	base.fRayZ = 0.1f;

	base.fDetSX = 0.0f;
	base.fDetSY = -300.0f;
	base.fDetSZ = -50.0f;

	base.fDetUX = 0.0f;
	base.fDetUY = 1.0f;
	base.fDetUZ = 0.0f;

	base.fDetVX = 0.0f;
	base.fDetVY = 0.0f;
	base.fDetVZ = 1.0f;

	SPar3DProjection angle[dims.iProjAngles];

	cudaPitchedPtr volData; // pitch, ptr, xsize, ysize

	volData = allocateVolumeData(dims);

	cudaPitchedPtr projData; // pitch, ptr, xsize, ysize

	projData = allocateProjectionData(dims);

	unsigned int ix = 500,iy = 500;

	float* buf = new float[dims.iProjU*dims.iProjV];

	float* slice = new float[dims.iVolX*dims.iVolY];
	for (int i = 0; i < dims.iVolX*dims.iVolY; ++i)
		slice[i] = 1.0f;

	for (unsigned int a = 0; a < 241; a += dims.iProjAngles) {

		zeroProjectionData(projData, dims);

		for (int y = 0; y < iy; y += dims.iVolY) {
			for (int x = 0; x < ix; x += dims.iVolX) { 

				timeval st;
				tic(st);

				for (int z = 0; z < dims.iVolZ; ++z) {
//					char sfn[256];
//					sprintf(sfn, "/home/wpalenst/projects/cone_simulation/phantom_4096/mouse_fem_phantom_%04d.png", 30+z);
//					float* slice = loadSubImage(sfn, x, y, dims.iVolX, dims.iVolY);

					cudaPitchedPtr ptr;
					ptr.ptr = slice;
					ptr.pitch = dims.iVolX*sizeof(float);
					ptr.xsize = dims.iVolX*sizeof(float);
					ptr.ysize = dims.iVolY;
					cudaExtent extentS;
					extentS.width = dims.iVolX*sizeof(float);
					extentS.height = dims.iVolY;
					extentS.depth = 1;

					cudaPos sp = { 0, 0, 0 };
					cudaPos dp = { 0, 0, z };
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
//					delete[] slice;
				}

				printf("Load: %f\n", toc(st));

#if 0

	cudaPos zp = { 0, 0, 0 };

	cudaPitchedPtr t;
	t.ptr = new float[1024*1024];
	t.pitch = 1024*4;
	t.xsize = 1024*4;
	t.ysize = 1024;

	cudaMemcpy3DParms p;
	p.srcArray = 0;
	p.srcPos = zp;
	p.srcPtr = volData;
	p.extent = extentS;
	p.dstArray = 0;
	p.dstPtr = t;
	p.dstPos = zp;
	p.kind = cudaMemcpyDeviceToHost;
	cudaError err = cudaMemcpy3D(&p);
	assert(!err);

	char fn[32];
	sprintf(fn, "t%d%d.png", x / dims.iVolX, y / dims.iVolY);
	saveImage(fn, 1024, 1024, (float*)t.ptr);
	saveImage("s.png", 4096, 4096, slice);
	delete[] (float*)t.ptr;
#endif


#define ROTATE0(name,i,alpha) do { angle[i].f##name##X = base.f##name##X * cos(alpha) - base.f##name##Y * sin(alpha); angle[i].f##name##Y = base.f##name##X * sin(alpha) + base.f##name##Y * cos(alpha); angle[i].f##name##Z = base.f##name##Z; } while(0)
#define SHIFT(name,i,x,y) do { angle[i].f##name##X += x; angle[i].f##name##Y += y; } while(0)
				for (int i = 0; i < dims.iProjAngles; ++i) {
					ROTATE0(Ray, i, (a+i)*.8*M_PI/180);
					ROTATE0(DetS, i, (a+i)*.8*M_PI/180);
					ROTATE0(DetU, i, (a+i)*.8*M_PI/180);
					ROTATE0(DetV, i, (a+i)*.8*M_PI/180);


//					SHIFT(Src, i, (-x+1536), (-y+1536));
//					SHIFT(DetS, i, (-x+1536), (-y+1536));
				}
#undef ROTATE0
#undef SHIFT
				tic(st);

				astraCUDA3d::Par3DFP(volData, projData, dims, angle, 1.0f);

				printf("FP: %f\n", toc(st));

			}
		}
		for (unsigned int aa = 0; aa < dims.iProjAngles; ++aa) {
			for (unsigned int v = 0; v < dims.iProjV; ++v)
				cudaMemcpy(buf+v*dims.iProjU, ((float*)projData.ptr)+(v*dims.iProjAngles+aa)*(projData.pitch/sizeof(float)), dims.iProjU*sizeof(float), cudaMemcpyDeviceToHost);

			char fname[32];
			sprintf(fname, "proj%03d.png", a+aa);
			saveImage(fname, dims.iProjV, dims.iProjU, buf, 0.0f, 1000.0f);
		}
	}

	delete[] buf;

}
#endif
