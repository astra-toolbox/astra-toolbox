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
#include "astra/cuda/3d/cone_bp_internal.h"

#include <cstdio>
#include <cassert>

static const unsigned int g_volBlockZ = 8;
static const unsigned int g_anglesPerBlock = astraCUDA3d::g_cuda3d_blockSlices; // TODO: clean up
static const unsigned int g_volBlockX = 32;
static const unsigned int g_volBlockY = 15;

// TODO: decouple g_MaxAngles and g_anglesPerBlock
// MaxAngles should become buffer size, and AnglesPerBlock the thread block size.
// MaxAngles should then probably be an integer multiple of AnglesPerBlock.
static const unsigned g_MaxAngles = g_anglesPerBlock;

static const unsigned int g_anglesPerPrepareBlock = 16;
static const unsigned int g_detBlockU = 32;
static const unsigned int g_detBlockV = 32;

#include "voxel3d.inl"

namespace astraCUDA3d {

__constant__ DevConeParams gC_C[g_MaxAngles];

using TransferConstantsBuffer = TransferConstantsBuffer_t<DevConeParams>;


template<Cuda3DProjectionKernel KERNEL, unsigned int ZSIZE>
__global__ void dev_cone_BP(float* D_volData, unsigned int volPitch,
                            cudaTextureObject_t tex,
                            int angleCount,
                            const astraCUDA3d::SDimensions3D dims,
                            float fOutputScale)
{
	const int X = blockIdx.x * blockDim.x + threadIdx.x;
	const int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= dims.iVolX)
		return;
	if (Y >= dims.iVolY)
		return;

	const int startZ = blockIdx.z * ZSIZE;
	const float fX = X - 0.5f*dims.iVolX + 0.5f;
	const float fY = Y - 0.5f*dims.iVolY + 0.5f;
	const float fZ = startZ - 0.5f*dims.iVolZ + 0.5f;

	float Z[ZSIZE];
	for(int i=0; i < ZSIZE; i++)
		Z[i] = 0.0f;


	{
		for (int angle = 0; angle < angleCount; ++angle)
		{
			float4 fCu  = gC_C[angle].fNumU;
			float4 fCv  = gC_C[angle].fNumV;
			float4 fCd  = gC_C[angle].fDen;

			float fUNum = fCu.w + fX * fCu.x + fY * fCu.y + fZ * fCu.z;
			float fVNum = fCv.w + fX * fCv.x + fY * fCv.y + fZ * fCv.z;
			float fDen  = ((KERNEL == ker3d_fdk_weighting) ? 1.0f : fCd.w) + fX * fCd.x + fY * fCd.y + fZ * fCd.z;

			float fU,fV, fr;

			for (int idx = 0; idx < ZSIZE; idx++)
			{
				fr = __fdividef(1.0f, fDen);
				fU = fUNum * fr;
				fV = fVNum * fr;
				float fVal = tex2DLayered<float>(tex, fU, fV, angle);
				if (KERNEL == ker3d_2d_weighting)
					Z[idx] += fr*fVal;
				else
					Z[idx] += fr*fr*fVal;

				fUNum += fCu.z;
				fVNum += fCv.z;
				fDen  += fCd.z;
			}
		}
	}

	int endZ = ZSIZE;
	if (endZ > dims.iVolZ - startZ)
		endZ = dims.iVolZ - startZ;

	for(int i=0; i < endZ; i++)
		D_volData[(size_t)((startZ+i)*dims.iVolY+Y)*volPitch+X] += Z[i] * fOutputScale;
} //End kernel



// supersampling version
__global__ void dev_cone_BP_SS(float* D_volData, unsigned int volPitch, cudaTextureObject_t tex, int angleCount, const SDimensions3D dims, int iRaysPerVoxelDim, unsigned int volBlockZ, float fOutputScale)
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
			float4 fCu  = gC_C[angle].fNumU;
			float4 fCv  = gC_C[angle].fNumV;
			float4 fCd  = gC_C[angle].fDen;

			float fXs = fX;
			for (int iSubX = 0; iSubX < iRaysPerVoxelDim; ++iSubX) {
			float fYs = fY;
			for (int iSubY = 0; iSubY < iRaysPerVoxelDim; ++iSubY) {
			float fZs = fZ;
			for (int iSubZ = 0; iSubZ < iRaysPerVoxelDim; ++iSubZ) {

				const float fUNum = fCu.w + fXs * fCu.x + fYs * fCu.y + fZs * fCu.z;
				const float fVNum = fCv.w + fXs * fCv.x + fYs * fCv.y + fZs * fCv.z;
				const float fDen  = fCd.w + fXs * fCd.x + fYs * fCd.y + fZs * fCd.z;

				const float fr = __fdividef(1.0f, fDen);
				const float fU = fUNum * fr;
				const float fV = fVNum * fr;

				fVal += tex2DLayered<float>(tex, fU, fV, angle) * fr * fr;

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

bool ConeBP_t::transferConstants(const SConeProjection* angles, unsigned int iProjAngles, const SProjectorParams3D& params,
                       TransferConstantsBuffer& buf, cudaStream_t stream)
{
	DevConeParams *p = &(std::get<0>(buf.d))[0];

	// We need three things in the kernel:
	// projected coordinates of pixels on the detector:

	// u: || (x-s) v (s-d) || / || u v (s-x) ||
	// v: -|| u (x-s) (s-d) || / || u v (s-x) ||

	// ray density weighting factor for the adjoint
	// || u v (s-d) ||^2 / ( |cross(u,v)| * || u v (s-x) ||^2 )

	// FDK weighting factor
	// ( || u v s || / || u v (s-x) || ) ^ 2

	// Since u and v are ratios with the same denominator, we have
	// a degree of freedom to scale the denominator. We use that to make
	// the square of the denominator equal to the relevant weighting factor.


	// We use an event to assure that the previous transferConstants has completed before
	// re-using the buffer. (Even if it is very unlikely that it hasn't.)
	bool ok = checkCuda(cudaStreamWaitEvent(stream, buf.event, 0), "transferConstants wait");

	for (unsigned int i = 0; i < iProjAngles; ++i) {
		Vec3 u(angles[i].fDetUX, angles[i].fDetUY, angles[i].fDetUZ);
		Vec3 v(angles[i].fDetVX, angles[i].fDetVY, angles[i].fDetVZ);
		Vec3 s(angles[i].fSrcX, angles[i].fSrcY, angles[i].fSrcZ);
		Vec3 d(angles[i].fDetSX, angles[i].fDetSY, angles[i].fDetSZ);



		double fScale;
		if (params.projKernel == ker3d_fdk_weighting) {
			// goal: 1/fDen = || u v s || / || u v (s-x) ||
			// fDen = || u v (s-x) || / || u v s ||
			// i.e., scale = 1 / || u v s ||

			fScale = 1.0 / det3(u, v, s);
		} else if (params.projKernel == ker3d_2d_weighting) {
			// We set the weights here to approximate the adjoint
			// of a 2d fanbeam kernel. To be used when only
			// operating on a single slice.
			// fDen = |cross(u,e_v)| || u v (s-x) || / || u v (s-d) ||
			Vec3 ev(0, 0, 1);
			fScale = scaled_cross3(u,ev,Vec3(params.volScale.fX,params.volScale.fY,params.volScale.fZ)).norm() / det3(u, v, s-d);

		} else {
			// goal: 1/fDen^2 = || u v (s-d) ||^2 / ( |cross(u,v)| * || u v (s-x) ||^2 )
			// fDen = ( sqrt(|cross(u,v)|) * || u v (s-x) || ) / || u v (s-d) || 
			// i.e. scale = sqrt(|cross(u,v)|) / || u v (s-d) ||


			// NB: for cross(u,v) we invert the volume scaling (for the voxel
			// size normalization) to get the proper dimensions for
			// the scaling of the adjoint

			fScale = sqrt(scaled_cross3(u,v,Vec3(params.volScale.fX,params.volScale.fY,params.volScale.fZ)).norm()) / det3(u, v, s-d);
		}

		p[i].fNumU.w = fScale * det3(s,v,d);
		p[i].fNumU.x = fScale * det3x(v,s-d);
		p[i].fNumU.y = fScale * det3y(v,s-d);
		p[i].fNumU.z = fScale * det3z(v,s-d);
		p[i].fNumV.w = -fScale * det3(s,u,d);
		p[i].fNumV.x = -fScale * det3x(u,s-d);
		p[i].fNumV.y = -fScale * det3y(u,s-d);
		p[i].fNumV.z = -fScale * det3z(u,s-d);
		p[i].fDen.w = fScale * det3(u, v, s); // == 1.0 for FDK
		p[i].fDen.x = -fScale * det3x(u, v);
		p[i].fDen.y = -fScale * det3y(u, v);
		p[i].fDen.z = -fScale * det3z(u, v);
	}

	ok &= checkCuda(cudaMemcpyToSymbolAsync(gC_C, p, iProjAngles*sizeof(DevConeParams), 0, cudaMemcpyHostToDevice, stream), "transferConstants transfer");

	ok &= checkCuda(cudaEventRecord(buf.event, stream), "transferConstants event");

	return ok;
}

__global__ void dev_cone_prepare(void* D_outData, const void* D_projData, unsigned int projPitchOut, unsigned int projPitchIn, unsigned int startAngle, const SDimensions3D bufdims, const SDimensions3D dims)
{
	const float* projData = (const float*)D_projData;
	float* outData = (float*)D_outData;
	int angle = blockIdx.z * blockDim.y + threadIdx.y;
	if (angle >= bufdims.iProjAngles)
		return;
	if (startAngle + angle >= dims.iProjAngles)
		return;

	const int detectorU = blockIdx.x * blockDim.x + threadIdx.x;
	const int startDetectorV = blockIdx.y * g_detBlockV;
	int endDetectorV = startDetectorV + g_detBlockV;
	if (endDetectorV > dims.iProjV)
		endDetectorV = dims.iProjV;

	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)
	{
		outData[(angle*bufdims.iProjV+detectorV)*projPitchOut+detectorU] = projData[(size_t)(detectorV*dims.iProjAngles+startAngle+angle)*projPitchIn+detectorU];
	}
}

bool ConeBP_t::prepareProjections(cudaPitchedPtr D_outData, cudaPitchedPtr D_projData,
                                  const SConeProjection* D_geom,
                                  unsigned int startAngle,
                                  const SDimensions3D& bufdims, const SDimensions3D& dims,
                                  const SProjectorParams3D& params,
                                  cudaStream_t stream)
{
	dim3 dimBlock(g_detBlockU, g_anglesPerPrepareBlock);
	dim3 dimGrid = computeGrid3Ceil(dims.iProjU, g_detBlockU,
	                                dims.iProjV, g_detBlockV,
	                                bufdims.iProjAngles, g_anglesPerPrepareBlock);

	int projPitchOut = D_outData.pitch/sizeof(float);
	int projPitchIn = D_projData.pitch/sizeof(float);

	dev_cone_prepare<<<dimGrid, dimBlock, 0, stream>>>(D_outData.ptr, D_projData.ptr, projPitchOut, projPitchIn, startAngle, bufdims, dims);

	return true;
}

bool ConeBP_t::dispatchKernel_BP(float *D_volumeData, unsigned int volPitch, cudaTextureObject_t D_texObj, unsigned int angleCount, const SDimensions3D& dims, const SProjectorParams3D& params, cudaStream_t stream)
{
	dim3 dimBlock(g_volBlockX, g_volBlockY);

	dim3 dimGrid = computeGrid3Ceil(dims.iVolX, g_volBlockX,
	                                dims.iVolY, g_volBlockY,
	                                dims.iVolZ, g_volBlockZ);

	if (params.projKernel == ker3d_fdk_weighting) {
		if (dims.iVolZ == 1) {
			dev_cone_BP<ker3d_fdk_weighting, 1><<<dimGrid, dimBlock, 0, stream>>>(D_volumeData, volPitch, D_texObj, angleCount, dims, params.fOutputScale);
		} else {
			dev_cone_BP<ker3d_fdk_weighting, g_volBlockZ><<<dimGrid, dimBlock, 0, stream>>>(D_volumeData, volPitch, D_texObj, angleCount, dims, params.fOutputScale);
		}
	} else if (params.projKernel == ker3d_2d_weighting) {
		if (dims.iVolZ == 1) {
			dev_cone_BP<ker3d_2d_weighting, 1><<<dimGrid, dimBlock, 0, stream>>>(D_volumeData, volPitch, D_texObj, angleCount, dims, params.fOutputScale);
		} else {
			// This combination doesn't necessarily make sense
			dev_cone_BP<ker3d_2d_weighting, g_volBlockZ><<<dimGrid, dimBlock, 0, stream>>>(D_volumeData, volPitch, D_texObj, angleCount, dims, params.fOutputScale);
		}
	} else if (params.iRaysPerVoxelDim == 1) {
		if (dims.iVolZ == 1) {
			dev_cone_BP<ker3d_default, 1><<<dimGrid, dimBlock, 0, stream>>>(D_volumeData, volPitch, D_texObj, angleCount, dims, params.fOutputScale);
		} else {
			dev_cone_BP<ker3d_default, g_volBlockZ><<<dimGrid, dimBlock, 0, stream>>>(D_volumeData, volPitch, D_texObj, angleCount, dims, params.fOutputScale);
		}
	} else {
		dev_cone_BP_SS<<<dimGrid, dimBlock, 0, stream>>>(D_volumeData, volPitch, D_texObj, angleCount, dims, params.iRaysPerVoxelDim, g_volBlockZ, params.fOutputScale);
	}
	return true;
}




bool ConeBP(cudaPitchedPtr D_volumeData,
            cudaPitchedPtr D_projData,
            const SDimensions3D& dims, const std::vector<SConeProjection>& angles,
            const SProjectorParams3D& params)
{
	ConeBP_t C;
	return C.project(D_volumeData, D_projData, ConeBP_t::MODE_BACKWARD, false, dims, angles, params);
}

}
