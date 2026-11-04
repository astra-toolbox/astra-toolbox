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

static const unsigned int g_volBlockZ = 8;
static const unsigned int g_anglesPerBlock = astraCUDA3d::g_cuda3d_blockSlices; // TODO: clean up
static const unsigned int g_volBlockX = 32;
static const unsigned int g_volBlockY = 15;

static const unsigned g_MaxAngles = g_anglesPerBlock;


#include "voxel3d.inl"

namespace astraCUDA3d {

struct DevPar3DParams {
	float4 fNumU;
	float4 fNumV;
};

__constant__ DevPar3DParams gC_C[g_MaxAngles];
__constant__ float gC_scale[g_MaxAngles];

using TransferConstantsBuffer = TransferConstantsBuffer_t<DevPar3DParams, float>;

class Par3DBP_t : public VoxelDrivenProjector3D<TransferConstantsBuffer, SPar3DProjection> {
public:
	virtual bool transferConstants(const SPar3DProjection* angles,
	                               unsigned int iProjAngles,
	                               const SProjectorParams3D& params,
	                               TransferConstantsBuffer& buf,
	                               cudaStream_t stream) override;

	virtual bool dispatchKernel_BP(float *D_volData, unsigned int volPitch,
	                               cudaTextureObject_t tex,
	                               unsigned int angleCount,
	                               const SDimensions3D& dims,
	                               const SProjectorParams3D& params,
	                               cudaStream_t stream) override;
};


template<unsigned int ZSIZE>
__global__ void dev_par3D_BP(float* D_volData, unsigned int volPitch, cudaTextureObject_t tex, int angleCount, const SDimensions3D dims, float fOutputScale)
{
	const int X = blockIdx.x * blockDim.x + threadIdx.x;
	const int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= dims.iVolX)
		return;
	if (Y >= dims.iVolY)
		return;

	const int startZ = blockIdx.z * ZSIZE;

	float fX = X - 0.5f*dims.iVolX + 0.5f;
	float fY = Y - 0.5f*dims.iVolY + 0.5f;
	float fZ = startZ - 0.5f*dims.iVolZ + 0.5f;

	float Z[ZSIZE];
	for(int i=0; i < ZSIZE; i++)
		Z[i] = 0.0f;

	{
		for (int angle = 0; angle < angleCount; ++angle)
		{

			float4 fCu = gC_C[angle].fNumU;
			float4 fCv = gC_C[angle].fNumV;
			float fS = gC_scale[angle];

			float fU = fCu.w + fX * fCu.x + fY * fCu.y + fZ * fCu.z;
			float fV = fCv.w + fX * fCv.x + fY * fCv.y + fZ * fCv.z;

			for (int idx = 0; idx < ZSIZE; ++idx) {

				float fVal = tex2DLayered<float>(tex, fU, fV, angle);
				Z[idx] += fVal * fS;

				fU += fCu.z;
				fV += fCv.z;
			}

		}
	}

	int endZ = ZSIZE;
	if (endZ > dims.iVolZ - startZ)
		endZ = dims.iVolZ - startZ;

	for(int i=0; i < endZ; i++)
		D_volData[(size_t)((startZ+i)*dims.iVolY+Y)*volPitch+X] += Z[i] * fOutputScale;
}

// supersampling version
__global__ void dev_par3D_BP_SS(float* D_volData, unsigned int volPitch, cudaTextureObject_t tex, int angleCount, const SDimensions3D dims, int iRaysPerVoxelDim, unsigned int volBlockZ, float fOutputScale)
{
	// TO TRY: precompute part of detector intersection formulas in shared mem?
	// TO TRY: inner loop over z, gather ray values in shared mem

	const int X = blockIdx.x * blockDim.x + threadIdx.x;
	const int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= dims.iVolX)
		return;
	if (Y >= dims.iVolY)
		return;

	const int startZ = blockIdx.z * volBlockZ;
	int endZ = startZ + volBlockZ;
	if (endZ > dims.iVolZ)
		endZ = dims.iVolZ;

	float fX = X - 0.5f*dims.iVolX + 0.5f - 0.5f + 0.5f/iRaysPerVoxelDim;
	float fY = Y - 0.5f*dims.iVolY + 0.5f - 0.5f + 0.5f/iRaysPerVoxelDim;
	float fZ = startZ - 0.5f*dims.iVolZ + 0.5f - 0.5f + 0.5f/iRaysPerVoxelDim;

	const float fSubStep = 1.0f/iRaysPerVoxelDim;

	fOutputScale /= (iRaysPerVoxelDim*iRaysPerVoxelDim*iRaysPerVoxelDim);


	for (int Z = startZ; Z < endZ; ++Z, fZ += 1.0f)
	{

		float fVal = 0.0f;

		for (int angle = 0; angle < angleCount; ++angle)
		{
			float4 fCu = gC_C[angle].fNumU;
			float4 fCv = gC_C[angle].fNumV;
			float fS = gC_scale[angle];

			float fXs = fX;
			for (int iSubX = 0; iSubX < iRaysPerVoxelDim; ++iSubX) {
			float fYs = fY;
			for (int iSubY = 0; iSubY < iRaysPerVoxelDim; ++iSubY) {
			float fZs = fZ;
			for (int iSubZ = 0; iSubZ < iRaysPerVoxelDim; ++iSubZ) {

				const float fU = fCu.w + fXs * fCu.x + fYs * fCu.y + fZs * fCu.z;
				const float fV = fCv.w + fXs * fCv.x + fYs * fCv.y + fZs * fCv.z;

				fVal += tex2DLayered<float>(tex, fU, fV, angle) * fS;
				fZs += fSubStep;
			}
			fYs += fSubStep;
			}
			fXs += fSubStep;
			}

		}

		D_volData[(size_t)(Z*dims.iVolY+Y)*volPitch+X] += fVal * fOutputScale;
	}

}


bool Par3DBP_t::transferConstants(const SPar3DProjection* angles, unsigned int iProjAngles, const SProjectorParams3D& params, TransferConstantsBuffer& buf, cudaStream_t stream)
{
	DevPar3DParams *p = &(std::get<0>(buf.d))[0];
	float *s = &(std::get<1>(buf.d))[0];

	// We use an event to assure that the previous transferConstants has completed before
	// re-using the buffer. (Even if it is very unlikely that it hasn't.)
	bool ok = checkCuda(cudaStreamWaitEvent(stream, buf.event, 0), "transferConstants wait");

	for (unsigned int i = 0; i < iProjAngles; ++i) {
		Vec3 u(angles[i].fDetUX, angles[i].fDetUY, angles[i].fDetUZ);
		Vec3 v(angles[i].fDetVX, angles[i].fDetVY, angles[i].fDetVZ);
		Vec3 r(angles[i].fRayX, angles[i].fRayY, angles[i].fRayZ);
		Vec3 d(angles[i].fDetSX, angles[i].fDetSY, angles[i].fDetSZ);

		double fDen = det3(r,u,v);
		p[i].fNumU.x = -det3x(r,v) / fDen;
		p[i].fNumU.y = -det3y(r,v) / fDen;
		p[i].fNumU.z = -det3z(r,v) / fDen;
		p[i].fNumU.w = -det3(r,d,v) / fDen;
		p[i].fNumV.x = det3x(r,u) / fDen;
		p[i].fNumV.y = det3y(r,u) / fDen;
		p[i].fNumV.z = det3z(r,u) / fDen;
		p[i].fNumV.w = det3(r,d,u) / fDen;

		if (params.projKernel == ker3d_2d_weighting) {
			// We set the scale here to approximate the adjoint
			// of a 2d parallel beam kernel. To be used when only
			// operating on a single slice.
			Vec3 ev(0, 0, 1);
			s[i] = 1.0 / scaled_cross3(u,ev,Vec3(params.volScale.fX,params.volScale.fY,params.volScale.fZ)).norm();
		} else {
			s[i] = 1.0 / scaled_cross3(u,v,Vec3(params.volScale.fX,params.volScale.fY,params.volScale.fZ)).norm();
		}
	}

	ok &= checkCuda(cudaMemcpyToSymbolAsync(gC_C, p, iProjAngles*sizeof(DevPar3DParams), 0, cudaMemcpyHostToDevice, stream), "transferConstants transfer C");
	ok &= checkCuda(cudaMemcpyToSymbolAsync(gC_scale, s, iProjAngles*sizeof(float), 0, cudaMemcpyHostToDevice, stream), "transferConstants transfer scale");

	ok &= checkCuda(cudaEventRecord(buf.event, stream), "transferConstants event");

	return ok;
}

bool Par3DBP_t::dispatchKernel_BP(float *D_volumeData, unsigned int volPitch, cudaTextureObject_t D_texObj, unsigned int angleCount, const SDimensions3D& dims, const SProjectorParams3D& params, cudaStream_t stream)
{
	dim3 dimBlock(g_volBlockX, g_volBlockY);
	dim3 dimGrid = computeGrid3Ceil(dims.iVolX, g_volBlockX,
	                                dims.iVolY, g_volBlockY,
	                                dims.iVolZ, g_volBlockZ);

	if (params.iRaysPerVoxelDim == 1) {
		if (dims.iVolZ == 1) {
			dev_par3D_BP<1><<<dimGrid, dimBlock, 0, stream>>>(D_volumeData, volPitch, D_texObj, angleCount, dims, params.fOutputScale);
		} else {
			dev_par3D_BP<g_volBlockZ><<<dimGrid, dimBlock, 0, stream>>>(D_volumeData, volPitch, D_texObj, angleCount, dims, params.fOutputScale);
		}
	} else {
		dev_par3D_BP_SS<<<dimGrid, dimBlock, 0, stream>>>(D_volumeData, volPitch, D_texObj, angleCount, dims, params.iRaysPerVoxelDim, g_volBlockZ, params.fOutputScale);
	}
	return true;
}

bool Par3DBP(cudaPitchedPtr D_volumeData,
             cudaPitchedPtr D_projData,
             const SDimensions3D& dims, const std::vector<SPar3DProjection>& angles,
             const SProjectorParams3D& params)
{
	Par3DBP_t P;
	return P.project(D_volumeData, D_projData, Par3DBP_t::MODE_BACKWARD, false, dims, angles, params);
}

}
