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

static texture3D gT_par3DProjTexture;

namespace astraCUDA3d {

#define ZSIZE 6
static const unsigned int g_volBlockZ = ZSIZE;

static const unsigned int g_anglesPerBlock = 32;
static const unsigned int g_volBlockX = 16;
static const unsigned int g_volBlockY = 32;

static const unsigned g_MaxAngles = 1024;

struct DevPar3DParams {
	float4 fNumU;
	float4 fNumV;
};

__constant__ DevPar3DParams gC_C[g_MaxAngles];
__constant__ float gC_scale[g_MaxAngles];


static bool bindProjDataTexture(const cudaArray* array)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	gT_par3DProjTexture.addressMode[0] = cudaAddressModeBorder;
	gT_par3DProjTexture.addressMode[1] = cudaAddressModeBorder;
	gT_par3DProjTexture.addressMode[2] = cudaAddressModeBorder;
	gT_par3DProjTexture.filterMode = cudaFilterModeLinear;
	gT_par3DProjTexture.normalized = false;

	cudaBindTextureToArray(gT_par3DProjTexture, array, channelDesc);

	// TODO: error value?

	return true;
}


__global__ void dev_par3D_BP(void* D_volData, unsigned int volPitch, int startAngle, int angleOffset, const SDimensions3D dims, float fOutputScale)
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

	float fX = X - 0.5f*dims.iVolX + 0.5f;
	float fY = Y - 0.5f*dims.iVolY + 0.5f;
	float fZ = startZ - 0.5f*dims.iVolZ + 0.5f;

	float Z[ZSIZE];
	for(int i=0; i < ZSIZE; i++)
		Z[i] = 0.0f;

	{
		float fAngle = startAngle + angleOffset + 0.5f;

		for (int angle = startAngle; angle < endAngle; ++angle, fAngle += 1.0f)
		{

			float4 fCu = gC_C[angle].fNumU;
			float4 fCv = gC_C[angle].fNumV;
			float fS = gC_scale[angle];

			float fU = fCu.w + fX * fCu.x + fY * fCu.y + fZ * fCu.z;
			float fV = fCv.w + fX * fCv.x + fY * fCv.y + fZ * fCv.z;

			for (int idx = 0; idx < ZSIZE; ++idx) {

				float fVal = tex3D(gT_par3DProjTexture, fU, fAngle, fV);
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
		volData[((startZ+i)*dims.iVolY+Y)*volPitch+X] += Z[i] * fOutputScale;
}

// supersampling version
__global__ void dev_par3D_BP_SS(void* D_volData, unsigned int volPitch, int startAngle, int angleOffset, const SDimensions3D dims, int iRaysPerVoxelDim, float fOutputScale)
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

				fVal += tex3D(gT_par3DProjTexture, fU, fAngle, fV) * fS;
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

bool transferConstants(const SPar3DProjection* angles, unsigned int iProjAngles, const SProjectorParams3D& params)
{
	DevPar3DParams *p = new DevPar3DParams[iProjAngles];
	float *s = new float[iProjAngles];

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

		s[i] = 1.0 / scaled_cross3(u,v,Vec3(params.fVolScaleX,params.fVolScaleY,params.fVolScaleZ)).norm();
	}

	cudaMemcpyToSymbol(gC_C, p, iProjAngles*sizeof(DevPar3DParams), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(gC_scale, s, iProjAngles*sizeof(float), 0, cudaMemcpyHostToDevice);

	delete[] p;
	delete[] s;

	return true;
}

bool Par3DBP_Array(cudaPitchedPtr D_volumeData,
                   cudaArray *D_projArray,
                   const SDimensions3D& dims, const SPar3DProjection* angles,
                   const SProjectorParams3D& params)
{
	bindProjDataTexture(D_projArray);

	float fOutputScale = params.fOutputScale * params.fVolScaleX * params.fVolScaleY * params.fVolScaleZ;

	for (unsigned int th = 0; th < dims.iProjAngles; th += g_MaxAngles) {
		unsigned int angleCount = g_MaxAngles;
		if (th + angleCount > dims.iProjAngles)
			angleCount = dims.iProjAngles - th;

		bool ok = transferConstants(angles, angleCount, params);
		if (!ok)
			return false;

		dim3 dimBlock(g_volBlockX, g_volBlockY);

		dim3 dimGrid(((dims.iVolX+g_volBlockX-1)/g_volBlockX)*((dims.iVolY+g_volBlockY-1)/g_volBlockY), (dims.iVolZ+g_volBlockZ-1)/g_volBlockZ);

		// timeval t;
		// tic(t);

		for (unsigned int i = 0; i < angleCount; i += g_anglesPerBlock) {
			// printf("Calling BP: %d, %dx%d, %dx%d to %p\n", i, dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y, (void*)D_volumeData.ptr); 
			if (params.iRaysPerVoxelDim == 1)
				dev_par3D_BP<<<dimGrid, dimBlock>>>(D_volumeData.ptr, D_volumeData.pitch/sizeof(float), i, th, dims, fOutputScale);
			else
				dev_par3D_BP_SS<<<dimGrid, dimBlock>>>(D_volumeData.ptr, D_volumeData.pitch/sizeof(float), i, th, dims, params.iRaysPerVoxelDim, fOutputScale);
		}

		cudaTextForceKernelsCompletion();

		angles = angles + angleCount;
		// printf("%f\n", toc(t));

	}

	return true;
}

bool Par3DBP(cudaPitchedPtr D_volumeData,
            cudaPitchedPtr D_projData,
            const SDimensions3D& dims, const SPar3DProjection* angles,
            const SProjectorParams3D& params)
{
	// transfer projections to array

	cudaArray* cuArray = allocateProjectionArray(dims);
	transferProjectionsToArray(D_projData, cuArray, dims);

	bool ret = Par3DBP_Array(D_volumeData, cuArray, dims, angles, params);

	cudaFreeArray(cuArray);

	return ret;
}


}
