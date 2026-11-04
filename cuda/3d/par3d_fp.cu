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
#include "astra/cuda/3d/proj3d.h"

#include <cstdio>
#include <cassert>

static const unsigned int g_anglesPerBlock = 15;
static const unsigned int g_blockSlices = astraCUDA3d::g_cuda3d_blockSlices; // TODO:clean up
static const unsigned int g_detBlockU = 32;
static const unsigned int g_detBlockV = 15;

static const unsigned g_MaxAngles = 1024;


#include "ray3d.inl"


namespace astraCUDA3d {

// TODO: We can remove RayZ if we need to (by dividing RayX and RayY by it)
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
__constant__ int gC_index[g_MaxAngles];



using TransferConstantsBuffer = TransferConstantsBuffer_t<float>;

class Par3DFP_t : public RayDrivenProjector3D<TransferConstantsBuffer, SPar3DProjection> {
public:
	virtual bool sortProjections(const SDimensions3D& dims,
	                             const std::vector<SPar3DProjection> &angles,
	                             std::vector<SPar3DProjection>& anglesX,
	                             std::vector<SPar3DProjection>& anglesY,
	                             std::vector<SPar3DProjection>& anglesZ,
	                             std::vector<int>& indicesX,
	                             std::vector<int>& indicesY,
	                             std::vector<int>& indicesZ) override;
	virtual bool transferConstants(const SPar3DProjection* angles, const int* indices,
	                               unsigned int iProjAngles,
	                               TransferConstantsBuffer& buf, cudaStream_t stream) override;
	virtual bool dispatchKernel(float *D_projData, unsigned int projPitch,
	                            cudaTextureObject_t tex,
	                            float fStartSlice,
	                            const SDimensions3D& dims,
	                            const SDimensions3D& blockDims,
	                            const SProjectorParams3D& params,
	                            int majordim,
	                            cudaStream_t stream) override;
};

bool Par3DFP_t::transferConstants(const SPar3DProjection* angles, const int* indices, unsigned int iProjAngles,
                                  TransferConstantsBuffer& buf, cudaStream_t stream)
{
	float* tmp = &(std::get<0>(buf.d))[0];

	// We use an event to assure that the previous transferConstants has completed before
	// re-using the buffer. (Even if it is very unlikely that it hasn't.)
	bool ok = checkCuda(cudaStreamWaitEvent(stream, buf.event, 0), "transferConstants wait");

#define TRANSFER_TO_CONSTANT(name) do { for (unsigned int i = 0; i < iProjAngles; ++i) tmp[i] = angles[i].f##name ; ok &= checkCuda(cudaMemcpyToSymbolAsync(gC_##name, tmp, iProjAngles*sizeof(float), 0, cudaMemcpyHostToDevice, stream), "transferConstants transfer"); } while (0)

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

	// TODO: If we ever move the temp buffers into pinned memory for
	// true async transfer, also use extra buffer for the indices.
	ok &= checkCuda(cudaMemcpyToSymbolAsync(gC_index, indices, iProjAngles * sizeof(int), 0, cudaMemcpyHostToDevice, stream), "transferConstants transfer indices");

	ok &= checkCuda(cudaEventRecord(buf.event, stream), "transferConstants event");

	return true;
}


// TODO: De-duplicate these kernels (regular, super-sampling, SumSqW) in a
// way that doesn't cause the cuda compiler to require more registers for
// them

template<class SCALE>
__global__ void par3D_FP_t(float* D_projData, unsigned int projPitch,
                           cudaTextureObject_t tex,
                           float fStartSlice,
                           const SDimensions3D blockDims, unsigned int totalAngles,
                           unsigned int detBlockV, unsigned int blockSlices,
                           SCALE sc)
{
	int angle = blockIdx.y * blockDim.y + threadIdx.y;
	if (angle >= blockDims.iProjAngles)
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

	// Get original index of this projection (for in the output array)
	angle = gC_index[angle];

	const int detectorU = blockIdx.x * blockDim.x + threadIdx.x;
	if (detectorU >= blockDims.iProjU)
		return;

	const float ax = fRayX / fRayZ;
	const float ay = fRayY / fRayZ;
	const float fDistCorr = sc.scale(ax, ay);

	const int startDetectorV = blockIdx.z * detBlockV;
	int endDetectorV = startDetectorV + detBlockV;
	if (endDetectorV > blockDims.iProjV)
		endDetectorV = blockDims.iProjV;

	int sliceCount = blockSlices;
	if (sliceCount > blockDims.iVolZ)
		sliceCount = blockDims.iVolZ;

	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)
	{
		/* Trace ray in direction Ray to (detectorU,detectorV) from  */
		/* Z = 0 to Z = sliceCount                            */

		const float fDetX = fDetSX + detectorU*fDetUX + detectorV*fDetVX;
		const float fDetY = fDetSY + detectorU*fDetUY + detectorV*fDetVY;
		const float fDetZ = fDetSZ + detectorU*fDetUZ + detectorV*fDetVZ;

		/*        (x)   (ax)       (bx)    */
		/* ray:   (y) = (ay) * z + (by)    */
		/*        (z)   ( 1)       ( 0)    */

		const float bx = fDetX - ax * fDetZ;
		const float by = fDetY - ay * fDetZ;

		float fVal = 0.0f;

		float fx = ax * fStartSlice + bx + 0.5f*blockDims.iVolX - 0.5f + 0.5f;
		float fy = ay * fStartSlice + by + 0.5f*blockDims.iVolY - 0.5f + 0.5f;

		for (int s = 0; s < sliceCount; ++s)
		{
			fVal += tex2DLayered<float>(tex, fx, fy, s);
			fx += ax;
			fy += ay;
		}

		fVal *= fDistCorr;

		D_projData[(size_t)(detectorV*totalAngles+angle)*projPitch+detectorU] += fVal;
	}
}

// Supersampling version
__global__ void par3D_FP_SS_t(float* D_projData, unsigned int projPitch,
                              cudaTextureObject_t tex,
                              float fStartSlice,
                              const SDimensions3D blockDims, unsigned int totalAngles,
                              int iRaysPerDetDim,
                              unsigned int detBlockV, unsigned int blockSlices,
                              SCALE_NONCUBE sc)
{
	int angle = blockIdx.y * blockDim.y + threadIdx.y;
	if (angle >= blockDims.iProjAngles)
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

	// Get original index of this projection (for in the output array)
	angle = gC_index[angle];

	const int detectorU = blockIdx.x * blockDim.x + threadIdx.x;
	if (detectorU >= blockDims.iProjU)
		return;

	const float ax = fRayX / fRayZ;
	const float ay = fRayY / fRayZ;
	const float fDistCorr = sc.scale(ax, ay);

	const int startDetectorV = blockIdx.z * detBlockV;
	int endDetectorV = startDetectorV + detBlockV;
	if (endDetectorV > blockDims.iProjV)
		endDetectorV = blockDims.iProjV;

	int sliceCount = blockSlices;
	if (sliceCount > blockDims.iVolZ)
		sliceCount = blockDims.iVolZ;

	const float fSubStep = 1.0f/iRaysPerDetDim;

	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)
	{

		float fV = 0.0f;

		float fdU = detectorU - 0.5f + 0.5f*fSubStep;
		for (int iSubU = 0; iSubU < iRaysPerDetDim; ++iSubU, fdU+=fSubStep) {
		float fdV = detectorV - 0.5f + 0.5f*fSubStep;
		for (int iSubV = 0; iSubV < iRaysPerDetDim; ++iSubV, fdV+=fSubStep) {

		/* Trace ray in direction Ray to (detectorU,detectorV) from  */
		/* Z = 0 to Z = sliceCount                            */

		const float fDetX = fDetSX + fdU*fDetUX + fdV*fDetVX;
		const float fDetY = fDetSY + fdU*fDetUY + fdV*fDetVY;
		const float fDetZ = fDetSZ + fdU*fDetUZ + fdV*fDetVZ;

		/*        (x)   (ax)       (bx)    */
		/* ray:   (y) = (ay) * z + (by)    */
		/*        (z)   ( 1)       ( 0)    */

		const float bx = fDetX - ax * fDetZ;
		const float by = fDetY - ay * fDetZ;

		float fVal = 0.0f;

		float fx = ax * fStartSlice + bx + 0.5f*blockDims.iVolX - 0.5f + 0.5f;
		float fy = ay * fStartSlice + by + 0.5f*blockDims.iVolY - 0.5f + 0.5f;

		for (int s = 0; s < sliceCount; ++s)
		{
			fVal += tex2DLayered<float>(tex, fx, fy, s);
			fx += ax;
			fy += ay;
		}

		fV += fVal;

		}
		}

		fV *= fDistCorr;
		D_projData[(size_t)(detectorV*totalAngles+angle)*projPitch+detectorU] += fV / (iRaysPerDetDim * iRaysPerDetDim);
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

__global__ void par3D_FP_SumSqW_t(float* D_projData, unsigned int projPitch,
                                  float fStartSlice,
                                  const SDimensions3D blockDims, unsigned int totalAngles,
                                  unsigned int detBlockV, unsigned int blockSlices,
                                  SCALE_NONCUBE sc)
{
	int angle = blockIdx.y * blockDim.y + threadIdx.y;
	if (angle >= blockDims.iProjAngles)
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

	// Get original index of this projection (for in the output array)
	angle = gC_index[angle];

	const int detectorU = blockIdx.x * blockDim.x + threadIdx.x;
	if (detectorU >= blockDims.iProjU)
		return;

	const float ax = fRayX / fRayZ;
	const float ay = fRayY / fRayZ;
	const float fDistCorr = sc.scale(ax, ay);

	const int startDetectorV = blockIdx.z * detBlockV;
	int endDetectorV = startDetectorV + detBlockV;
	if (endDetectorV > blockDims.iProjV)
		endDetectorV = blockDims.iProjV;

	int sliceCount = blockSlices;
	if (sliceCount > blockDims.iVolZ)
		sliceCount = blockDims.iVolZ;

	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)
	{
		/* Trace ray in direction Ray to (detectorU,detectorV) from  */
		/* Z = 0 to Z = sliceCount                            */

		const float fDetX = fDetSX + detectorU*fDetUX + detectorV*fDetVX;
		const float fDetY = fDetSY + detectorU*fDetUY + detectorV*fDetVY;
		const float fDetZ = fDetSZ + detectorU*fDetUZ + detectorV*fDetVZ;

		/*        (x)   (ax)       (bx)    */
		/* ray:   (y) = (ay) * z + (by)    */
		/*        (z)   ( 1)       ( 0)    */

		const float bx = fDetX - ax * fDetZ;
		const float by = fDetY - ay * fDetZ;

		float fVal = 0.0f;

		float fx = ax * fStartSlice + bx + 0.5f*blockDims.iVolX - 0.5f + 0.5f;
		float fy = ay * fStartSlice + by + 0.5f*blockDims.iVolY - 0.5f + 0.5f;

		for (int s = 0; s < sliceCount; ++s)
		{
			fVal += dirWeights(fx, blockDims.iVolX) * dirWeights(fy, blockDims.iVolY);
			fx += ax;
			fy += ay;
		}

		fVal *= fDistCorr * fDistCorr;
		D_projData[(size_t)(detectorV*totalAngles+angle)*projPitch+detectorU] += fVal;
	}
}

// Supersampling version
// TODO

bool Par3DFP_t::sortProjections(const SDimensions3D& dims, const std::vector<SPar3DProjection>& angles,
                               std::vector<SPar3DProjection>& anglesX,
                               std::vector<SPar3DProjection>& anglesY,
                               std::vector<SPar3DProjection>& anglesZ,
                               std::vector<int>& indicesX,
                               std::vector<int>& indicesY,
                               std::vector<int>& indicesZ)
{
	assert(dims.iProjAngles == angles.size());

	anglesX.clear();
	anglesX.reserve(dims.iProjAngles);
	anglesY.clear();
	anglesY.reserve(dims.iProjAngles);
	anglesZ.clear();
	anglesZ.reserve(dims.iProjAngles);
	indicesX.clear();
	indicesX.reserve(dims.iProjAngles);
	indicesY.clear();
	indicesY.reserve(dims.iProjAngles);
	indicesZ.clear();
	indicesZ.reserve(dims.iProjAngles);

	for (unsigned int i = 0; i < dims.iProjAngles; ++i) {
		float dX = fabsf(angles[i].fRayX);
		float dY = fabsf(angles[i].fRayY);
		float dZ = fabsf(angles[i].fRayZ);

		if (dX >= dY && dX >= dZ) {
			// X
			// X <- Y, Y <- Z, Z <- X
			SPar3DProjection a;
			a.fRayX = angles[i].fRayY;
			a.fRayY = angles[i].fRayZ;
			a.fRayZ = angles[i].fRayX;
			a.fDetSX = angles[i].fDetSY;
			a.fDetSY = angles[i].fDetSZ;
			a.fDetSZ = angles[i].fDetSX;
			a.fDetUX = angles[i].fDetUY;
			a.fDetUY = angles[i].fDetUZ;
			a.fDetUZ = angles[i].fDetUX;
			a.fDetVX = angles[i].fDetVY;
			a.fDetVY = angles[i].fDetVZ;
			a.fDetVZ = angles[i].fDetVX;
			//fprintf(stderr, "Index %d -> X\n", i);
			anglesX.push_back(a);
			indicesX.push_back(i);
		} else if (dY >= dX && dY >= dZ) {
			// Y
			// X <- X, Y <- Z, Z <- Y
			SPar3DProjection a;
			a.fRayX = angles[i].fRayX;
			a.fRayY = angles[i].fRayZ;
			a.fRayZ = angles[i].fRayY;
			a.fDetSX = angles[i].fDetSX;
			a.fDetSY = angles[i].fDetSZ;
			a.fDetSZ = angles[i].fDetSY;
			a.fDetUX = angles[i].fDetUX;
			a.fDetUY = angles[i].fDetUZ;
			a.fDetUZ = angles[i].fDetUY;
			a.fDetVX = angles[i].fDetVX;
			a.fDetVY = angles[i].fDetVZ;
			a.fDetVZ = angles[i].fDetVY;
			anglesY.push_back(a);
			indicesY.push_back(i);
		} else {
			// Z
			// No coordinate permutations
			anglesZ.push_back(angles[i]);
			indicesZ.push_back(i);
		}
	}
	return true;
}


bool Par3DFP_t::dispatchKernel(float *D_projData, unsigned int projPitch,
	                       cudaTextureObject_t D_texObj,
	                       float fStartSlice,
	                       const SDimensions3D& dims,
	                       const SDimensions3D& blockDims,
	                       const SProjectorParams3D& params,
	                       int majordim,
                               cudaStream_t stream)
{
	bool cube = isVolCube(params);
	SCALE_CUBE scube = getCubeScaling(params);
	SCALE_NONCUBE snoncube = getGenericScaling(params, majordim);

	dim3 dimBlock(g_detBlockU, g_anglesPerBlock);
	dim3 dimGrid = computeGrid3Ceil(blockDims.iProjU, g_detBlockU,
	                                blockDims.iProjAngles, g_anglesPerBlock,
	                                blockDims.iProjV, g_detBlockV);

	switch (params.projKernel) {
	case ker3d_default:
	case ker3d_2d_weighting:
	case ker3d_matched_bp:
		if (params.iRaysPerDetDim == 1) {
			if (cube)
				par3D_FP_t<<<dimGrid, dimBlock, 0, stream>>>(D_projData, projPitch, D_texObj, fStartSlice, blockDims, dims.iProjAngles, g_detBlockV, g_blockSlices, scube);
			else
				par3D_FP_t<<<dimGrid, dimBlock, 0, stream>>>(D_projData, projPitch, D_texObj, fStartSlice, blockDims, dims.iProjAngles, g_detBlockV, g_blockSlices, snoncube);
		} else {
			par3D_FP_SS_t<<<dimGrid, dimBlock, 0, stream>>>(D_projData, projPitch, D_texObj, fStartSlice, blockDims, dims.iProjAngles, params.iRaysPerDetDim, g_detBlockV, g_blockSlices, snoncube);
		}
		return true;
	case ker3d_sum_square_weights:
		if (params.iRaysPerDetDim == 1) {
			par3D_FP_SumSqW_t<<<dimGrid, dimBlock, 0, stream>>>(D_projData, projPitch, fStartSlice, blockDims, dims.iProjAngles, g_detBlockV, g_blockSlices, snoncube);
		} else {
			// TODO
			return false;
		}
		return true;
	default:
		return false;
	}
}

bool Par3DFP(cudaPitchedPtr D_volumeData,
             cudaPitchedPtr D_projData,
             const SDimensions3D& dims, const std::vector<SPar3DProjection>& angles,
             const SProjectorParams3D& params)
{
	Par3DFP_t P;
	return P.project(D_volumeData, D_projData, Par3DFP_t::MODE_FORWARD, dims, angles, params);
}


bool Par3DFP_SumSqW(cudaPitchedPtr D_projData,
                    const SDimensions3D& dims, const std::vector<SPar3DProjection>& angles,
                    const SProjectorParams3D& params)
{
	Par3DFP_t P;
	cudaPitchedPtr dummy = { nullptr, 0, 0, 0 };
	return P.project(dummy, D_projData, Par3DFP_t::MODE_FW_NOINPUT, dims, angles, params);
}






}
