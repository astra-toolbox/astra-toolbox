/*
-----------------------------------------------------------------------
Copyright: 2021-2022, CWI, Amsterdam
           2021-2022, University of Cambridge

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

-----------------------------------------------------------------------
*/


#include "astra/cuda/3d/util3d.h"
#include "astra/cuda/3d/dims3d.h"

#include "astra/Logging.h"

#include <cstdio>
#include <cassert>
#include <iostream>
#include <list>

#include <cuda.h>

namespace astraCUDA3d {

static const unsigned int g_volBlockZ = 6;

static const unsigned int g_anglesPerBlock = 32;
static const unsigned int g_volBlockX = 16;
static const unsigned int g_volBlockY = 32;

// must be divisible by anglesPerBlock
static const unsigned g_MaxAngles = 768;

struct DevCylConeParamsBP {
	float4 fCa;
	float4 fCb;
	float4 fCc;
	float3 fSmC;
	float fDetUT;
	float fInvNormV;
	float fR;
};

__constant__ DevCylConeParamsBP gC_C[g_MaxAngles];

//__launch_bounds__(32*16, 4)
template<bool FDKWEIGHT, unsigned int ZSIZE>
__global__ void dev_cylcone_BP(void* D_volData, unsigned int volPitch,
                            cudaTextureObject_t tex,
                            int startAngle, int angleOffset,
                            const astraCUDA3d::SDimensions3D dims,
                            float fOutputScale,
                            float fvx, float fvy, float fvz )
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
	const float fX = fvx * (X - 0.5f*dims.iVolX + 0.5f);
	const float fY = fvy * (Y - 0.5f*dims.iVolY + 0.5f);
	const float fZ = fvz * (startZ - 0.5f*dims.iVolZ + 0.5f);

	float Z[ZSIZE];
	for(int i=0; i < ZSIZE; i++)
		Z[i] = 0.0f;


	{
		float fAngle = startAngle + angleOffset + 0.5f;

		for (int angle = startAngle; angle < endAngle; ++angle, fAngle += 1.0f)
		{
			const float4 fCa  = gC_C[angle].fCa;
			const float4 fCb  = gC_C[angle].fCb;
			const float4 fCc  = gC_C[angle].fCc;

			float ra = fCa.w + fX * fCa.x + fY * fCa.y + fZ * fCa.z;
			float rb = fCb.w + fX * fCb.x + fY * fCb.y + fZ * fCb.z;
			float rc = fCc.w + fX * fCc.x + fY * fCc.y + fZ * fCc.z;

			const float sa = gC_C[angle].fSmC.x;
			const float sb = gC_C[angle].fSmC.y;
			const float sc = gC_C[angle].fSmC.z;

			const float R = gC_C[angle].fR;

			for (int idx = 0; idx < ZSIZE; idx++)
			{
				const float A = ra * ra + rb * rb;
				const float B = ra * sa + rb * sb; // removed factor 2
				const float C = sa * sa + sb * sb - R * R;
				const float D = B * B - A * C; // removed factor 4
				if (D < 0)
					continue;
				const float t = (-B + sqrt(D)) / A; // cancelled removed factor 2 ( = sqrt(4))

				const float theta = asin((t * rb + sb) / R) / gC_C[angle].fDetUT;
				const float v = (t * rc + sc) * gC_C[angle].fInvNormV; // TODO: fold invnormv into rc, sc, dc

				const float fU = theta + 0.5f * dims.iProjU; // TODO: CHECKME
				const float fV = v + 0.5f * dims.iProjV; // TODO: CHECKME

				//printf("%f,%f,%f %f,%f,%f %f %f theta: %f v: %f U: %f V: %f (%d %d)\n", fX, fY, fZ, A, B, C, D, t, theta, v, fU, fV, dims.iProjU, dims.iProjV);
//printf("%f %f %f\n", (ra * ra + rb * rb + rc * rc), t, (ra * ra + rb * rb + rc * rc) / t);
				float fVal = tex3D<float>(tex, fU, fAngle, fV);
				// The ray density is 1 / t^2
				// (t is scaled in transferConstants for this)
				Z[idx] += fVal * t * t;


				ra += fvz * fCa.z;
				rb += fvz * fCb.z;
				rc += fvz * fCc.z;
			}
		}
	}

	int endZ = ZSIZE;
	if (endZ > dims.iVolZ - startZ)
		endZ = dims.iVolZ - startZ;

	for(int i=0; i < endZ; i++)
		volData[((startZ+i)*dims.iVolY+Y)*volPitch+X] += Z[i] * fOutputScale;
} //End kernel



bool transferConstants_conecylbp(const SCylConeProjection* angles, unsigned int iProjAngles, const SProjectorParams3D& params)
{
	DevCylConeParamsBP *p = new DevCylConeParamsBP[iProjAngles];

	for (unsigned int i = 0; i < iProjAngles; ++i) {
		double fRadius = angles[i].fDetR;
		Vec3 u(angles[i].fDetUX, angles[i].fDetUY, angles[i].fDetUZ);
		Vec3 v(angles[i].fDetVX, angles[i].fDetVY, angles[i].fDetVZ);
		Vec3 s(angles[i].fSrcX, angles[i].fSrcY, angles[i].fSrcZ);
		Vec3 d(angles[i].fDetCX, angles[i].fDetCY, angles[i].fDetCZ);

		Vec3 cyla, cylb, cylc, cylaxis;
		getCylConeAxes(angles[i], cyla, cylb, cylc, cylaxis);

		//Vec3 cyla = -cross3(u, v) * (fRadius / (u.norm() * v.norm())); 
		//Vec3 cylc = d - cyla;
		//Vec3 cylb = u * (fRadius / u.norm());

		Vec3 cyla_n = cyla * (1.0 / cyla.norm());
		Vec3 cylb_n = cylb * (1.0 / cylb.norm());
		Vec3 cylc_n = cylc * (1.0 / cylc.norm());
		Vec3 v_n = v * (1.0 / v.norm());

		// Without this scaling factor, the ray density in a point x is given
		// by 1 / (| cross(u,v | * t^2 ). We rescale all components of C here
		// so that the ray density is instead given by 1 / t^2.

		double fScale = sqrt(cross3(u, v).norm());

		p[i].fCa.x = cyla_n.x * fScale;
		p[i].fCa.y = cyla_n.y * fScale;
		p[i].fCa.z = cyla_n.z * fScale;
		p[i].fCb.x = cylb_n.x * fScale;
		p[i].fCb.y = cylb_n.y * fScale;
		p[i].fCb.z = cylb_n.z * fScale;
		p[i].fCc.x = v_n.x * fScale;
		p[i].fCc.y = v_n.y * fScale;
		p[i].fCc.z = v_n.z * fScale;

		p[i].fCa.w = -s.dot(cyla_n) * fScale;
		p[i].fCb.w = -s.dot(cylb_n) * fScale;
		p[i].fCc.w = -s.dot(v_n) * fScale;

#if 1

		p[i].fSmC.x = (s - cylc).dot(cyla_n);
		p[i].fSmC.y = (s - cylc).dot(cylb_n);
		p[i].fSmC.z = (s - cylc).dot(v_n);
#endif
		
		// angular increment
		p[i].fDetUT = u.norm() / fRadius;
		p[i].fInvNormV = 1.0 / v.norm();

		p[i].fR = fRadius;

		// TODO: density weighting factor |u x v| missing!

		if (i == 0) {
			//fprintf(stderr, "%f,%f,%f,%f  %f,%f,%f,%f %f,%f,%f,%f  %f,%f,%f  %f %f\n", p[i].fCa.x,p[i].fCa.y, p[i].fCa.z, p[i].fCa.w, p[i].fCb.x,p[i].fCb.y, p[i].fCb.z, p[i].fCb.w, p[i].fCc.x,p[i].fCc.y, p[i].fCc.z, p[i].fCc.w,  p[i].fSmC.x,p[i].fSmC.y, p[i].fSmC.z, p[i].fDetUT, p[i].fInvNormV);
		}
	}





	// TODO: Check for errors
	cudaMemcpyToSymbol(gC_C, p, iProjAngles*sizeof(DevCylConeParamsBP), 0, cudaMemcpyHostToDevice);

	delete[] p;

	return true;
}


bool ConeCylBP_Array(cudaPitchedPtr D_volumeData,
                  cudaArray *D_projArray,
                  const SDimensions3D& dims, const SCylConeProjection* angles,
                  const SProjectorParams3D& params)
{
	cudaTextureObject_t D_texObj;
	if (!createTextureObject3D(D_projArray, D_texObj))
		return false;

	float fOutputScale;
	if (params.bFDKWeighting) {
		ASTRA_ERROR("FDK not supported for ConeCylBP");
		return false;
	} else {
		fOutputScale = params.fOutputScale * (params.fVolScaleX * params.fVolScaleY * params.fVolScaleZ);
	}

	bool ok = true;

	for (unsigned int th = 0; th < dims.iProjAngles; th += g_MaxAngles) {
		unsigned int angleCount = g_MaxAngles;
		if (th + angleCount > dims.iProjAngles)
			angleCount = dims.iProjAngles - th;

		ok = transferConstants_conecylbp(angles, angleCount, params);
		if (!ok)
			break;

		dim3 dimBlock(g_volBlockX, g_volBlockY);

		dim3 dimGrid(((dims.iVolX/1+g_volBlockX-1)/(g_volBlockX))*((dims.iVolY/1+1*g_volBlockY-1)/(1*g_volBlockY)), (dims.iVolZ+g_volBlockZ-1)/g_volBlockZ);

		// timeval t;
		// tic(t);

		for (unsigned int i = 0; i < angleCount; i += g_anglesPerBlock) {
		// printf("Calling BP: %d, %dx%d, %dx%d to %p\n", i, dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y, (void*)D_volumeData.ptr); 
			if (params.bFDKWeighting) {
				if (dims.iVolZ == 1) {
					dev_cylcone_BP<true, 1><<<dimGrid, dimBlock>>>(D_volumeData.ptr, D_volumeData.pitch/sizeof(float), D_texObj, i, th, dims, fOutputScale, params.fVolScaleX, params.fVolScaleY, params.fVolScaleZ);
				} else {
					dev_cylcone_BP<true, g_volBlockZ><<<dimGrid, dimBlock>>>(D_volumeData.ptr, D_volumeData.pitch/sizeof(float), D_texObj, i, th, dims, fOutputScale, params.fVolScaleX, params.fVolScaleY, params.fVolScaleZ);
				}
			} else if (params.iRaysPerVoxelDim == 1) {
				if (dims.iVolZ == 1) {
					dev_cylcone_BP<false, 1><<<dimGrid, dimBlock>>>(D_volumeData.ptr, D_volumeData.pitch/sizeof(float), D_texObj, i, th, dims, fOutputScale, params.fVolScaleX, params.fVolScaleY, params.fVolScaleZ);

				} else {
					dev_cylcone_BP<false, g_volBlockZ><<<dimGrid, dimBlock>>>(D_volumeData.ptr, D_volumeData.pitch/sizeof(float), D_texObj, i, th, dims, fOutputScale, params.fVolScaleX, params.fVolScaleY, params.fVolScaleZ);
				}
			} else {
				// TODO?
				assert(false);
			}
		}

		// TODO: Consider not synchronizing here, if possible.
		ok = checkCuda(cudaThreadSynchronize(), "cone_cyl_bp");
		if (!ok)
			break;

		angles = angles + angleCount;
		// printf("%f\n", toc(t));

	}

	cudaDestroyTextureObject(D_texObj);

	return ok;
}

bool ConeCylBP(cudaPitchedPtr D_volumeData,
            cudaPitchedPtr D_projData,
            const SDimensions3D& dims, const SCylConeProjection* angles,
            const SProjectorParams3D& params)
{
	// transfer projections to array

	cudaArray* cuArray = allocateProjectionArray(dims);
	transferProjectionsToArray(D_projData, cuArray, dims);

	bool ret = ConeCylBP_Array(D_volumeData, cuArray, dims, angles, params);

	cudaFreeArray(cuArray);

	return ret;
}


}
