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

#include "astra/cuda/3d/util3d.h"
#include "astra/cuda/3d/dims3d.h"

#include <cstdio>
#include <cassert>

namespace astraCUDA3d {

static const unsigned int g_anglesPerBlock = 4;

// thickness of the slices we're splitting the volume up into
static const unsigned int g_blockSlices = 4;
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


// x=0, y=1, z=2
struct DIR_X {
	__device__ float nSlices(const SDimensions3D& dims) const { return dims.iVolX; }
	__device__ float nDim1(const SDimensions3D& dims) const { return dims.iVolY; }
	__device__ float nDim2(const SDimensions3D& dims) const { return dims.iVolZ; }
	__device__ float c0(float x, float y, float z) const { return x; }
	__device__ float c1(float x, float y, float z) const { return y; }
	__device__ float c2(float x, float y, float z) const { return z; }
	__device__ float tex(cudaTextureObject_t tex, float f0, float f1, float f2) const { return tex3D<float>(tex, f0, f1, f2); }
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
	__device__ float tex(cudaTextureObject_t tex, float f0, float f1, float f2) const { return tex3D<float>(tex, f1, f0, f2); }
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
	__device__ float tex(cudaTextureObject_t tex, float f0, float f1, float f2) const { return tex3D<float>(tex, f1, f2, f0); }
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

using TransferConstantsBuffer = TransferConstantsBuffer_t<float>;

bool transferConstants(const SConeProjection* angles, unsigned int iProjAngles,
                       TransferConstantsBuffer& buf, cudaStream_t stream)
{
	float* tmp = &(std::get<0>(buf.d))[0];

	// We use an event to assure that the previous transferConstants has completed before
	// re-using the buffer. (Even if it is very unlikely that it hasn't.)
	bool ok = checkCuda(cudaStreamWaitEvent(stream, buf.event, 0), "transferConstants wait");

#define TRANSFER_TO_CONSTANT(name) do { for (unsigned int i = 0; i < iProjAngles; ++i) tmp[i] = angles[i].f##name ; ok &= checkCuda(cudaMemcpyToSymbolAsync(gC_##name, tmp, iProjAngles*sizeof(float), 0, cudaMemcpyHostToDevice, stream), "transferConstants transfer"); } while (0)

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

	ok &= checkCuda(cudaEventRecord(buf.event, stream), "transferConstants event");

	return ok;
}


	// threadIdx: x = ??? detector  (u?)
	//            y = relative angle

	// blockIdx:  x = ??? detector  (u+v?)
    //            y = angle block

template<class COORD, class SCALE>
__global__ void cone_FP_t(float* D_projData, unsigned int projPitch,
                          cudaTextureObject_t tex,
                          unsigned int startSlice,
                          unsigned int startAngle, unsigned int endAngle,
                          const SDimensions3D dims,
                          SCALE sc)
{
	COORD c;

	int angle = startAngle + blockIdx.y * g_anglesPerBlock + threadIdx.y;
	if (angle >= endAngle)
		return;

	const float fSrcX = gC_SrcX[angle];
	const float fSrcY = gC_SrcY[angle];
	const float fSrcZ = gC_SrcZ[angle];
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
	if (detectorU >= dims.iProjU)
		return;
	const int startDetectorV = (blockIdx.x/((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockV;
	int endDetectorV = startDetectorV + g_detBlockV;
	if (endDetectorV > dims.iProjV)
		endDetectorV = dims.iProjV;

	int endSlice = startSlice + g_blockSlices;
	if (endSlice > c.nSlices(dims))
		endSlice = c.nSlices(dims);

	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)
	{
		/* Trace ray from Src to (detectorU,detectorV) from */
		/* X = startSlice to X = endSlice                   */

		const float fDetX = fDetSX + detectorU*fDetUX + detectorV*fDetVX;
		const float fDetY = fDetSY + detectorU*fDetUY + detectorV*fDetVY;
		const float fDetZ = fDetSZ + detectorU*fDetUZ + detectorV*fDetVZ;

		/*        (x)   ( 1)       ( 0) */
		/* ray:   (y) = (ay) * x + (by) */
		/*        (z)   (az)       (bz) */

		const float a1 = (c.c1(fSrcX,fSrcY,fSrcZ) - c.c1(fDetX,fDetY,fDetZ)) / (c.c0(fSrcX,fSrcY,fSrcZ) - c.c0(fDetX,fDetY,fDetZ));
		const float a2 = (c.c2(fSrcX,fSrcY,fSrcZ) - c.c2(fDetX,fDetY,fDetZ)) / (c.c0(fSrcX,fSrcY,fSrcZ) - c.c0(fDetX,fDetY,fDetZ));
		const float b1 = c.c1(fSrcX,fSrcY,fSrcZ) - a1 * c.c0(fSrcX,fSrcY,fSrcZ);
		const float b2 = c.c2(fSrcX,fSrcY,fSrcZ) - a2 * c.c0(fSrcX,fSrcY,fSrcZ);

		const float fDistCorr = sc.scale(a1, a2);

		float fVal = 0.0f;

		float f0 = startSlice + 0.5f;
		float f1 = a1 * (startSlice - 0.5f*c.nSlices(dims) + 0.5f) + b1 + 0.5f*c.nDim1(dims) - 0.5f + 0.5f;
		float f2 = a2 * (startSlice - 0.5f*c.nSlices(dims) + 0.5f) + b2 + 0.5f*c.nDim2(dims) - 0.5f + 0.5f;

		for (int s = startSlice; s < endSlice; ++s)
		{
			fVal += c.tex(tex, f0, f1, f2);
			f0 += 1.0f;
			f1 += a1;
			f2 += a2;
		}

		fVal *= fDistCorr;

		D_projData[(size_t)(detectorV*dims.iProjAngles+angle)*projPitch+detectorU] += fVal;
	}
}

template<class COORD>
__global__ void cone_FP_SS_t(float* D_projData, unsigned int projPitch,
                             cudaTextureObject_t tex,
                             unsigned int startSlice,
                             unsigned int startAngle, unsigned int endAngle,
                             const SDimensions3D dims, int iRaysPerDetDim,
                             SCALE_NONCUBE sc)
{
	COORD c;

	int angle = startAngle + blockIdx.y * g_anglesPerBlock + threadIdx.y;
	if (angle >= endAngle)
		return;

	const float fSrcX = gC_SrcX[angle];
	const float fSrcY = gC_SrcY[angle];
	const float fSrcZ = gC_SrcZ[angle];
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
	if (detectorU >= dims.iProjU)
		return;
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
		/* Trace ray from Src to (detectorU,detectorV) from */
		/* X = startSlice to X = endSlice                   */

		float fV = 0.0f;

		float fdU = detectorU - 0.5f + 0.5f*fSubStep;
		for (int iSubU = 0; iSubU < iRaysPerDetDim; ++iSubU, fdU+=fSubStep) {
		float fdV = detectorV - 0.5f + 0.5f*fSubStep;
		for (int iSubV = 0; iSubV < iRaysPerDetDim; ++iSubV, fdV+=fSubStep) {

		const float fDetX = fDetSX + fdU*fDetUX + fdV*fDetVX;
		const float fDetY = fDetSY + fdU*fDetUY + fdV*fDetVY;
		const float fDetZ = fDetSZ + fdU*fDetUZ + fdV*fDetVZ;

		/*        (x)   ( 1)       ( 0) */
		/* ray:   (y) = (ay) * x + (by) */
		/*        (z)   (az)       (bz) */

		const float a1 = (c.c1(fSrcX,fSrcY,fSrcZ) - c.c1(fDetX,fDetY,fDetZ)) / (c.c0(fSrcX,fSrcY,fSrcZ) - c.c0(fDetX,fDetY,fDetZ));
		const float a2 = (c.c2(fSrcX,fSrcY,fSrcZ) - c.c2(fDetX,fDetY,fDetZ)) / (c.c0(fSrcX,fSrcY,fSrcZ) - c.c0(fDetX,fDetY,fDetZ));
		const float b1 = c.c1(fSrcX,fSrcY,fSrcZ) - a1 * c.c0(fSrcX,fSrcY,fSrcZ);
		const float b2 = c.c2(fSrcX,fSrcY,fSrcZ) - a2 * c.c0(fSrcX,fSrcY,fSrcZ);

		const float fDistCorr = sc.scale(a1, a2);

		float fVal = 0.0f;

		float f0 = startSlice + 0.5f;
		float f1 = a1 * (startSlice - 0.5f*c.nSlices(dims) + 0.5f) + b1 + 0.5f*c.nDim1(dims) - 0.5f + 0.5f;
		float f2 = a2 * (startSlice - 0.5f*c.nSlices(dims) + 0.5f) + b2 + 0.5f*c.nDim2(dims) - 0.5f + 0.5f;

		for (int s = startSlice; s < endSlice; ++s)
		{
			fVal += c.tex(tex, f0, f1, f2);
			f0 += 1.0f;
			f1 += a1;
			f2 += a2;
		}

		fVal *= fDistCorr;
		fV += fVal;

		}
		}

		D_projData[(size_t)(detectorV*dims.iProjAngles+angle)*projPitch+detectorU] += fV / (iRaysPerDetDim * iRaysPerDetDim);
	}
}


bool ConeFP_Array_internal(cudaPitchedPtr D_projData,
                  cudaTextureObject_t D_texObj,
                  const SDimensions3D& dims,
                  unsigned int angleCount, const SConeProjection* angles,
                  const SProjectorParams3D& params, cudaStream_t stream)
{
	dim3 dimBlock(g_detBlockU, g_anglesPerBlock); // region size, angles

	// Run over all angles, grouping them into groups of the same
	// orientation (roughly horizontal vs. roughly vertical).
	// Start a stream of grids for each such group.

	unsigned int blockStart = 0;
	unsigned int blockEnd = 0;
	int blockDirection = 0;

	bool cube = true;
	if (abs(params.volScale.fX / params.volScale.fY - 1.0) > 0.00001)
		cube = false;
	if (abs(params.volScale.fX / params.volScale.fZ - 1.0) > 0.00001)
		cube = false;

	SCALE_CUBE scube;
	scube.fOutputScale = params.fOutputScale * params.volScale.fX;

	SCALE_NONCUBE snoncubeX;
	float fS1 = params.volScale.fY / params.volScale.fX;
	snoncubeX.fScale1 = fS1 * fS1;
	float fS2 = params.volScale.fZ / params.volScale.fX;
	snoncubeX.fScale2 = fS2 * fS2;
	snoncubeX.fOutputScale = params.fOutputScale * params.volScale.fX;

	SCALE_NONCUBE snoncubeY;
	fS1 = params.volScale.fX / params.volScale.fY;
	snoncubeY.fScale1 = fS1 * fS1;
	fS2 = params.volScale.fZ / params.volScale.fY;
	snoncubeY.fScale2 = fS2 * fS2;
	snoncubeY.fOutputScale = params.fOutputScale * params.volScale.fY;

	SCALE_NONCUBE snoncubeZ;
	fS1 = params.volScale.fX / params.volScale.fZ;
	snoncubeZ.fScale1 = fS1 * fS1;
	fS2 = params.volScale.fY / params.volScale.fZ;
	snoncubeZ.fScale2 = fS2 * fS2;
	snoncubeZ.fOutputScale = params.fOutputScale * params.volScale.fZ;

	// timeval t;
	// tic(t);

	for (unsigned int a = 0; a <= angleCount; ++a) {
		int dir = -1;
		if (a != angleCount) {
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

		if (a == angleCount || dir != blockDirection) {
			// block done

			blockEnd = a;
			if (blockStart != blockEnd) {

				dim3 dimGrid(
				             ((dims.iProjU+g_detBlockU-1)/g_detBlockU)*((dims.iProjV+g_detBlockV-1)/g_detBlockV),
(blockEnd-blockStart+g_anglesPerBlock-1)/g_anglesPerBlock);

				// printf("angle block: %d to %d, %d (%dx%d, %dx%d)\n", blockStart, blockEnd, blockDirection, dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

				if (blockDirection == 0) {
					for (unsigned int i = 0; i < dims.iVolX; i += g_blockSlices)
						if (params.iRaysPerDetDim == 1)
							if (cube)
								cone_FP_t<DIR_X><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), D_texObj, i, blockStart, blockEnd, dims, scube);
							else
								cone_FP_t<DIR_X><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), D_texObj, i, blockStart, blockEnd, dims, snoncubeX);
						else
							cone_FP_SS_t<DIR_X><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), D_texObj, i, blockStart, blockEnd, dims, params.iRaysPerDetDim, snoncubeX);
				} else if (blockDirection == 1) {
					for (unsigned int i = 0; i < dims.iVolY; i += g_blockSlices)
						if (params.iRaysPerDetDim == 1)
							if (cube)
								cone_FP_t<DIR_Y><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), D_texObj, i, blockStart, blockEnd, dims, scube);
							else
								cone_FP_t<DIR_Y><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), D_texObj, i, blockStart, blockEnd, dims, snoncubeY);
						else
							cone_FP_SS_t<DIR_Y><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), D_texObj, i, blockStart, blockEnd, dims, params.iRaysPerDetDim, snoncubeY);
				} else if (blockDirection == 2) {
					for (unsigned int i = 0; i < dims.iVolZ; i += g_blockSlices)
						if (params.iRaysPerDetDim == 1)
							if (cube)
								cone_FP_t<DIR_Z><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), D_texObj, i, blockStart, blockEnd, dims, scube);
							else
								cone_FP_t<DIR_Z><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), D_texObj, i, blockStart, blockEnd, dims, snoncubeZ);
						else
							cone_FP_SS_t<DIR_Z><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), D_texObj, i, blockStart, blockEnd, dims, params.iRaysPerDetDim, snoncubeZ);
				}

			}

			blockDirection = dir;
			blockStart = a;
		}
	}

	// printf("%f\n", toc(t));

	return true;
}


bool ConeFP(cudaPitchedPtr D_volumeData,
            cudaPitchedPtr D_projData,
            const SDimensions3D& dims, const SConeProjection* angles,
            const SProjectorParams3D& params)
{
	TransferConstantsBuffer tcbuf(g_MaxAngles);

	cudaStream_t stream;
	if (!checkCuda(cudaStreamCreate(&stream), "ConeFP stream"))
		return false;

	// transfer volume to array
	cudaArray* cuArray = allocateVolumeArray(dims);
	if (!cuArray) {
		cudaStreamDestroy(stream);
		return false;
	}

	cudaTextureObject_t D_texObj;
	if (!createTextureObject3D(cuArray, D_texObj)) {
		cudaStreamDestroy(stream);
		cudaFreeArray(cuArray);
		return false;
	}

	if (!transferVolumeToArray(D_volumeData, cuArray, dims, stream)) {
		cudaDestroyTextureObject(D_texObj);
		cudaStreamDestroy(stream);
		cudaFreeArray(cuArray);
		return false;
	}

	bool ok = true;

	for (unsigned int iAngle = 0; iAngle < dims.iProjAngles; iAngle += g_MaxAngles) {
		unsigned int iEndAngle = iAngle + g_MaxAngles;
		if (iEndAngle >= dims.iProjAngles)
			iEndAngle = dims.iProjAngles;

		ok = transferConstants(angles + iAngle, iEndAngle - iAngle, tcbuf, stream);
		if (!ok)
			break;

		cudaPitchedPtr D_subprojData = D_projData;
		D_subprojData.ptr = (char*)D_projData.ptr + iAngle * D_projData.pitch;

		ok = ConeFP_Array_internal(D_subprojData, D_texObj,
		                            dims, iEndAngle - iAngle, angles + iAngle,
		                            params, stream);
		if (!ok)
			break;
	}

	ok &= checkCuda(cudaStreamSynchronize(stream), "ConeFP sync");

	cudaDestroyTextureObject(D_texObj);
	cudaFreeArray(cuArray);
	cudaStreamDestroy(stream);

	return ok;
}


}
