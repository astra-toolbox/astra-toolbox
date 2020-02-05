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

static texture3D gT_coneProjTexture;

namespace astraCUDA3d {

#define ZSIZE 6
static const unsigned int g_volBlockZ = ZSIZE;

static const unsigned int g_anglesPerBlock = 32;
static const unsigned int g_volBlockX = 16;
static const unsigned int g_volBlockY = 32;

static const unsigned g_MaxAngles = 1024;

struct DevConeParams {
	float4 fNumU;
	float4 fNumV;
	float4 fDen;
};

__constant__ DevConeParams gC_C[g_MaxAngles];

bool bindProjDataTexture(const cudaArray* array)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	gT_coneProjTexture.addressMode[0] = cudaAddressModeBorder;
	gT_coneProjTexture.addressMode[1] = cudaAddressModeBorder;
	gT_coneProjTexture.addressMode[2] = cudaAddressModeBorder;
	gT_coneProjTexture.filterMode = cudaFilterModeLinear;
	gT_coneProjTexture.normalized = false;

	cudaBindTextureToArray(gT_coneProjTexture, array, channelDesc);

	// TODO: error value?

	return true;
}


//__launch_bounds__(32*16, 4)
template<bool FDKWEIGHT>
__global__ void dev_cone_BP(void* D_volData, unsigned int volPitch, int startAngle,
                            int angleOffset, const astraCUDA3d::SDimensions3D dims,
                            float fOutputScale)
{
	float* volData = (float*)D_volData;

	int endAngle = startAngle + g_anglesPerBlock;
	if (endAngle > dims.iProjAngles - angleOffset)
		endAngle = dims.iProjAngles - angleOffset;

	// threadIdx: x = rel x
	//            y = rel y

	// blockIdx:  x = x + y
	//            y = z



	const int X = blockIdx.x % ((dims.iVolX+g_volBlockX-1)/g_volBlockX) * g_volBlockX + threadIdx.x;
	const int Y = blockIdx.x / ((dims.iVolX+g_volBlockX-1)/g_volBlockX) * g_volBlockY + threadIdx.y;

	if (X >= dims.iVolX)
		return;
	if (Y >= dims.iVolY)
		return;

	const int startZ = blockIdx.y * g_volBlockZ;
	const float fX = X - 0.5f*dims.iVolX + 0.5f;
	const float fY = Y - 0.5f*dims.iVolY + 0.5f;
	const float fZ = startZ - 0.5f*dims.iVolZ + 0.5f;

	float Z[ZSIZE];
	for(int i=0; i < ZSIZE; i++)
		Z[i] = 0.0f;


	{
		float fAngle = startAngle + angleOffset + 0.5f;

		for (int angle = startAngle; angle < endAngle; ++angle, fAngle += 1.0f)
		{
			float4 fCu  = gC_C[angle].fNumU;
			float4 fCv  = gC_C[angle].fNumV;
			float4 fCd  = gC_C[angle].fDen;

			float fUNum = fCu.w + fX * fCu.x + fY * fCu.y + fZ * fCu.z;
			float fVNum = fCv.w + fX * fCv.x + fY * fCv.y + fZ * fCv.z;
			float fDen  = (FDKWEIGHT ? 1.0f : fCd.w) + fX * fCd.x + fY * fCd.y + fZ * fCd.z;

			float fU,fV, fr;

			for (int idx = 0; idx < ZSIZE; idx++)
			{
				fr = __fdividef(1.0f, fDen);
				fU = fUNum * fr;
				fV = fVNum * fr;
				float fVal = tex3D(gT_coneProjTexture, fU, fAngle, fV);
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
		volData[((startZ+i)*dims.iVolY+Y)*volPitch+X] += Z[i] * fOutputScale;
} //End kernel



// supersampling version
__global__ void dev_cone_BP_SS(void* D_volData, unsigned int volPitch, int startAngle, int angleOffset, const SDimensions3D dims, int iRaysPerVoxelDim, float fOutputScale)
{
	float* volData = (float*)D_volData;

	int endAngle = startAngle + g_anglesPerBlock;
	if (endAngle > dims.iProjAngles - angleOffset)
		endAngle = dims.iProjAngles - angleOffset;

	// threadIdx: x = rel x
	//            y = rel y

	// blockIdx:  x = x + y
    //            y = z


	// TO TRY: precompute part of detector intersection formulas in shared mem?
	// TO TRY: inner loop over z, gather ray values in shared mem

	const int X = blockIdx.x % ((dims.iVolX+g_volBlockX-1)/g_volBlockX) * g_volBlockX + threadIdx.x;
	const int Y = blockIdx.x / ((dims.iVolX+g_volBlockX-1)/g_volBlockX) * g_volBlockY + threadIdx.y;

	if (X >= dims.iVolX)
		return;
	if (Y >= dims.iVolY)
		return;

	const int startZ = blockIdx.y * g_volBlockZ;
	int endZ = startZ + g_volBlockZ;
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
		float fAngle = startAngle + angleOffset + 0.5f;

		for (int angle = startAngle; angle < endAngle; ++angle, fAngle += 1.0f)
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

				const float fUNum = fCu.w + fX * fCu.x + fY * fCu.y + fZ * fCu.z;
				const float fVNum = fCv.w + fX * fCv.x + fY * fCv.y + fZ * fCv.z;
				const float fDen  = fCd.w + fX * fCd.x + fY * fCd.y + fZ * fCd.z;

				const float fr = __fdividef(1.0f, fDen);
				const float fU = fUNum * fr;
				const float fV = fVNum * fr;

				fVal += tex3D(gT_coneProjTexture, fU, fV, fAngle) * fr;

				fZs += fSubStep;
			}
			fYs += fSubStep;
			}
			fXs += fSubStep;
			}

		}

		volData[(Z*dims.iVolY+Y)*volPitch+X] += fVal * fOutputScale;
	}
}


bool transferConstants(const SConeProjection* angles, unsigned int iProjAngles, const SProjectorParams3D& params)
{
	DevConeParams *p = new DevConeParams[iProjAngles];

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


	for (unsigned int i = 0; i < iProjAngles; ++i) {
		Vec3 u(angles[i].fDetUX, angles[i].fDetUY, angles[i].fDetUZ);
		Vec3 v(angles[i].fDetVX, angles[i].fDetVY, angles[i].fDetVZ);
		Vec3 s(angles[i].fSrcX, angles[i].fSrcY, angles[i].fSrcZ);
		Vec3 d(angles[i].fDetSX, angles[i].fDetSY, angles[i].fDetSZ);



		double fScale;
		if (!params.bFDKWeighting) {
			// goal: 1/fDen^2 = || u v (s-d) ||^2 / ( |cross(u,v)| * || u v (s-x) ||^2 )
			// fDen = ( sqrt(|cross(u,v)|) * || u v (s-x) || ) / || u v (s-d) || 
			// i.e. scale = sqrt(|cross(u,v)|) * / || u v (s-d) ||


			// NB: for cross(u,v) we invert the volume scaling (for the voxel
			// size normalization) to get the proper dimensions for
			// the scaling of the adjoint

			fScale = sqrt(scaled_cross3(u,v,Vec3(params.fVolScaleX,params.fVolScaleY,params.fVolScaleZ)).norm()) / det3(u, v, s-d);
		} else {
			// goal: 1/fDen = || u v s || / || u v (s-x) ||
			// fDen = || u v (s-x) || / || u v s ||
			// i.e., scale = 1 / || u v s ||

			fScale = 1.0 / det3(u, v, s);
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

	// TODO: Check for errors
	cudaMemcpyToSymbol(gC_C, p, iProjAngles*sizeof(DevConeParams), 0, cudaMemcpyHostToDevice);

	delete[] p;

	return true;
}


bool ConeBP_Array(cudaPitchedPtr D_volumeData,
                  cudaArray *D_projArray,
                  const SDimensions3D& dims, const SConeProjection* angles,
                  const SProjectorParams3D& params)
{
	bindProjDataTexture(D_projArray);

	float fOutputScale;
	if (params.bFDKWeighting) {
		// NB: assuming cube voxels here
		fOutputScale = params.fOutputScale / (params.fVolScaleX);
	} else {
		fOutputScale = params.fOutputScale * (params.fVolScaleX * params.fVolScaleY * params.fVolScaleZ);
	}

	for (unsigned int th = 0; th < dims.iProjAngles; th += g_MaxAngles) {
		unsigned int angleCount = g_MaxAngles;
		if (th + angleCount > dims.iProjAngles)
			angleCount = dims.iProjAngles - th;

		bool ok = transferConstants(angles, angleCount, params);
		if (!ok)
			return false;

		dim3 dimBlock(g_volBlockX, g_volBlockY);

		dim3 dimGrid(((dims.iVolX/1+g_volBlockX-1)/(g_volBlockX))*((dims.iVolY/1+1*g_volBlockY-1)/(1*g_volBlockY)), (dims.iVolZ+g_volBlockZ-1)/g_volBlockZ);

		// timeval t;
		// tic(t);

		for (unsigned int i = 0; i < angleCount; i += g_anglesPerBlock) {
		// printf("Calling BP: %d, %dx%d, %dx%d to %p\n", i, dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y, (void*)D_volumeData.ptr); 
			if (params.bFDKWeighting)
				dev_cone_BP<true><<<dimGrid, dimBlock>>>(D_volumeData.ptr, D_volumeData.pitch/sizeof(float), i, th, dims, fOutputScale);
			else if (params.iRaysPerVoxelDim == 1)
				dev_cone_BP<false><<<dimGrid, dimBlock>>>(D_volumeData.ptr, D_volumeData.pitch/sizeof(float), i, th, dims, fOutputScale);
			else
				dev_cone_BP_SS<<<dimGrid, dimBlock>>>(D_volumeData.ptr, D_volumeData.pitch/sizeof(float), i, th, dims, params.iRaysPerVoxelDim, fOutputScale);
		}

		cudaTextForceKernelsCompletion();

		angles = angles + angleCount;
		// printf("%f\n", toc(t));

	}


	return true;
}

bool ConeBP(cudaPitchedPtr D_volumeData,
            cudaPitchedPtr D_projData,
            const SDimensions3D& dims, const SConeProjection* angles,
            const SProjectorParams3D& params)
{
	// transfer projections to array

	cudaArray* cuArray = allocateProjectionArray(dims);
	transferProjectionsToArray(D_projData, cuArray, dims);

	bool ret = ConeBP_Array(D_volumeData, cuArray, dims, angles, params);

	cudaFreeArray(cuArray);

	return ret;
}


}
