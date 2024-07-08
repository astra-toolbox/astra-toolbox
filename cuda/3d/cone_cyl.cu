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

// TODO: Change interpolation kernel to account for anisotropic voxels
// possible idea: for voxel width 1 and interpolation kernel width s,
// modf(coordinate,  detector index, frac)
// __saturatef(frac / s)
// __saturatef( (1 - frac) / s)    (NB: two separate saturates for L/R coeffs)
// (Also interesting: half-precision intrinsic __hmul_sat for saturating mul)
// Need to test impact of saturate vs branches (unnecessary reads vs
// unnecessary branching) with typical anisotropy scales.
//
// other idea:
// For integral ratios can also compute L/R detector indices and scale those.
// (Effect: left and right detector indices can then be equal, and their
//          coefficients would then sum to 1)
// But typical ratio doesn't appear to be integral.

#include <cstdio>
#include <cassert>
#include <iostream>
#include <list>

#include <cuda.h>
#include "astra/cuda/3d/util3d.h"

#ifdef STANDALONE
#include "testutil.h"
#endif

#include "astra/cuda/3d/dims3d.h"

namespace astraCUDA3d {

static const unsigned int g_anglesPerBlock = 4;

// thickness of the slices we're splitting the volume up into
static const unsigned int g_blockSlices = 16;
static const unsigned int g_detBlockU = 32;
static const unsigned int g_detBlockV = 16;

static const unsigned g_MaxAngles = 960;

struct DevCylConeParams {
	float3 fSrc;
	float3 fCylC;
	float3 fCylA;
	float3 fCylB;
	float3 fDetV;
	float fDetUT;
};

__constant__ DevCylConeParams gC_C[g_MaxAngles];



bool transferConstants(const SCylConeProjection* angles, unsigned int iProjAngles, const SProjectorParams3D& params)
{
	DevCylConeParams *p = new DevCylConeParams[iProjAngles];

	for (unsigned int i = 0; i < iProjAngles; ++i) {
		double fRadius = angles[i].fDetR;
		Vec3 u(angles[i].fDetUX, angles[i].fDetUY, angles[i].fDetUZ);
		Vec3 v(angles[i].fDetVX, angles[i].fDetVY, angles[i].fDetVZ);
		Vec3 s(angles[i].fSrcX, angles[i].fSrcY, angles[i].fSrcZ);
		Vec3 d(angles[i].fDetCX, angles[i].fDetCY, angles[i].fDetCZ);

		Vec3 cyla, cylb, cylc, cylaxis;
		getCylConeAxes(angles[i], cyla, cylb, cylc, cylaxis);

		// angular increment
		p[i].fDetUT = u.norm() / fRadius;

		p[i].fSrc.x =  s.x;
		p[i].fSrc.y =  s.y;
		p[i].fSrc.z =  s.z;
		p[i].fCylC.x = cylc.x;
		p[i].fCylC.y = cylc.y;
		p[i].fCylC.z = cylc.z;
		p[i].fCylA.x = cyla.x;
		p[i].fCylA.y = cyla.y;
		p[i].fCylA.z = cyla.z;
		p[i].fCylB.x = cylb.x;
		p[i].fCylB.y = cylb.y;
		p[i].fCylB.z = cylb.z;
		p[i].fDetV.x = v.x;
		p[i].fDetV.y = v.y;
		p[i].fDetV.z = v.z;
	}

	//fprintf(stderr, "%f %f %f  ===   %f %f %f    ===   %f %f %f   ===    %f %f %f   ===   %f %f %f   ==== %f\n", p[0].fSrc.x, p[0].fSrc.y, p[0].fSrc.z, p[0].fCylC.x, p[0].fCylC.y, p[0].fCylC.z, p[0].fCylA.x, p[0].fCylA.y, p[0].fCylA.z, p[0].fCylB.x, p[0].fCylB.y, p[0].fCylB.z, p[0].fDetV.x, p[0].fDetV.y, p[0].fDetV.z, p[0].fDetUT);

	if (!checkCuda(cudaMemcpyToSymbol(gC_C, p, iProjAngles*sizeof(DevCylConeParams), 0, cudaMemcpyHostToDevice), "curved_fp transferConstants memcpy")) {
		delete[] p;
		return false;
	}

	delete[] p;

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
	__device__ float tex(cudaTextureObject_t tex, float f0, float f1, float f2) const { return tex3D<float>(tex, f0, f1, f2); }
	__device__ float x(float f0, float f1, float f2) const { return f0; }
	__device__ float y(float f0, float f1, float f2) const { return f1; }
	__device__ float z(float f0, float f1, float f2) const { return f2; }
	__device__ int ix(int f0, int f1, int f2) const { return f0; }
	__device__ int iy(int f0, int f1, int f2) const { return f1; }
	__device__ int iz(int f0, int f1, int f2) const { return f2; }
	__device__ unsigned int stride0(const SDimensions3D& dims, unsigned int pitch) const { return 1; }
	__device__ unsigned int stride1(const SDimensions3D& dims, unsigned int pitch) const { return pitch; }
	__device__ unsigned int stride2(const SDimensions3D& dims, unsigned int pitch) const { return dims.iVolY * pitch; }
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
	__device__ int ix(int f0, int f1, int f2) const { return f1; }
	__device__ int iy(int f0, int f1, int f2) const { return f0; }
	__device__ int iz(int f0, int f1, int f2) const { return f2; }
	__device__ unsigned int stride0(const SDimensions3D& dims, unsigned int pitch) const { return pitch; }
	__device__ unsigned int stride1(const SDimensions3D& dims, unsigned int pitch) const { return 1; }
	__device__ unsigned int stride2(const SDimensions3D& dims, unsigned int pitch) const { return dims.iVolY * pitch; }
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
	__device__ int ix(int f0, int f1, int f2) const { return f1; }
	__device__ int iy(int f0, int f1, int f2) const { return f2; }
	__device__ int iz(int f0, int f1, int f2) const { return f0; }
	__device__ unsigned int stride0(const SDimensions3D& dims, unsigned int pitch) const { return dims.iVolY * pitch; }
	__device__ unsigned int stride1(const SDimensions3D& dims, unsigned int pitch) const { return 1; }
	__device__ unsigned int stride2(const SDimensions3D& dims, unsigned int pitch) const { return pitch; }
};


	// threadIdx: x = ??? detector  (u?)
	//            y = relative angle

	// blockIdx:  x = ??? detector  (u+v?)
    //            y = angle block


template<class COORD>
__global__ void cylcone_FP_t(float* D_projData, unsigned int projPitch,
                          cudaTextureObject_t tex,
                          unsigned int startSlice,
                          unsigned int startAngle, unsigned int endAngle,
                          const SDimensions3D dims,
                          float fR, // cylinder radius
                          float fvx, float fvy, float fvz, // voxel sizes
                          float fOutputScale
                         )
{
	COORD c;

	int angle = startAngle + blockIdx.y * g_anglesPerBlock + threadIdx.y;
	if (angle >= endAngle)
		return;

	const float fSrcX = gC_C[angle].fSrc.x;
	const float fSrcY = gC_C[angle].fSrc.y;
	const float fSrcZ = gC_C[angle].fSrc.z;
	const float fDetVX = gC_C[angle].fDetV.x;
	const float fDetVY = gC_C[angle].fDetV.y;
	const float fDetVZ = gC_C[angle].fDetV.z;
	const float fCylCX = gC_C[angle].fCylC.x;
	const float fCylCY = gC_C[angle].fCylC.y;
	const float fCylCZ = gC_C[angle].fCylC.z;

	// There's a bit of redundancy in these if we need to save space:
	const float fCylAX = gC_C[angle].fCylA.x;
	const float fCylAY = gC_C[angle].fCylA.y;
	const float fCylAZ = gC_C[angle].fCylA.z;
	const float fCylBX = gC_C[angle].fCylB.x;
	const float fCylBY = gC_C[angle].fCylB.y;
	const float fCylBZ = gC_C[angle].fCylB.z;
	const float fDetUT = gC_C[angle].fDetUT;

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

	const float fDetU = (2*detectorU - (int)dims.iProjU + 1) * 0.5f * fDetUT;
	float fsindu, fcosdu;
	sincosf(fDetU, &fsindu, &fcosdu);

	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)
	{
		const float fDetV = detectorV - 0.5f * dims.iProjV + 0.5f;
		/* Trace ray from Src to (detectorU,detectorV) from */
		/* X = startSlice to X = endSlice                   */

		const float fDetX = fCylCX + fCylAX*fcosdu + fCylBX*fsindu + fDetVX*fDetV;
		const float fDetY = fCylCY + fCylAY*fcosdu + fCylBY*fsindu + fDetVY*fDetV;
		const float fDetZ = fCylCZ + fCylAZ*fcosdu + fCylBZ*fsindu + fDetVZ*fDetV;

		//printf("%d %d -> %f %f %f\n", detectorU, detectorV, fDetX, fDetY, fDetZ);

		/*        (x)   ( 1)       ( 0) */
		/* ray:   (y) = (ay) * x + (by) */
		/*        (z)   (az)       (bz) */
		

		float a1 = (c.c1(fSrcX,fSrcY,fSrcZ) - c.c1(fDetX,fDetY,fDetZ)) / (c.c0(fSrcX,fSrcY,fSrcZ) - c.c0(fDetX,fDetY,fDetZ)) ;

		float a2 = (c.c2(fSrcX,fSrcY,fSrcZ) - c.c2(fDetX,fDetY,fDetZ)) / (c.c0(fSrcX,fSrcY,fSrcZ) - c.c0(fDetX,fDetY,fDetZ)) ;

		const float fDistCorr = sqrt(a1*a1 + a2*a2 + 1.0f) * c.c0(fvx, fvy, fvz);

		// a1, a2 are in global units above. Converted to texture steps below.

		a1 /= c.c1(fvx, fvy, fvz);
		a2 /= c.c2(fvx, fvy, fvz);

		const float b1 = c.c1(fSrcX,fSrcY,fSrcZ) / c.c1(fvx, fvy, fvz) - a1 * c.c0(fSrcX,fSrcY,fSrcZ);
		const float b2 = c.c2(fSrcX,fSrcY,fSrcZ) / c.c2(fvx, fvy, fvz) - a2 * c.c0(fSrcX,fSrcY,fSrcZ);

		a1 *= c.c0(fvx, fvy, fvz);
		a2 *= c.c0(fvx, fvy, fvz);


		float fVal = 0.0f;

		float f0 = startSlice + 0.5f;
		float f1 = a1 * (startSlice + (0.5 - 0.5f*c.nSlices(dims)) ) + b1 + 0.5f*c.nDim1(dims);
		float f2 = a2 * (startSlice + (0.5 - 0.5f*c.nSlices(dims)) ) + b2 + 0.5f*c.nDim2(dims);

		for (int s = startSlice; s < endSlice; ++s)
		{
//printf("F: %d %d,%d %f,%f,%f\n", angle, detectorV, detectorU, f0, f1, f2);
			// TODO: Move away from texture interpolation for (strongly) anisotropic pixels
			// See BP for coefficients
			fVal += c.tex(tex, f0, f1, f2);
			f0 += 1.0f;
			f1 += a1;
			f2 += a2;
		}

		fVal *= fDistCorr * fOutputScale;

		D_projData[(detectorV*dims.iProjAngles+angle)*projPitch+detectorU] += fVal;
	}
}


template<class COORD>
__global__ void cylcone_BP_t(float* D_volData, unsigned int volPitch,
                          cudaTextureObject_t tex,
                          unsigned int startSlice, unsigned int relAngle,
                          unsigned int startAngle, unsigned int endAngle,
                          const SDimensions3D dims,
                          float fR, // cylinder radius
                          float fvx, float fvy, float fvz // voxel sizes
                         )
{
	COORD c;

	int angle = startAngle + blockIdx.y * g_anglesPerBlock + threadIdx.y;
	if (angle >= endAngle)
		return;

	const float fSrcX = gC_C[angle].fSrc.x;
	const float fSrcY = gC_C[angle].fSrc.y;
	const float fSrcZ = gC_C[angle].fSrc.z;
	const float fDetVX = gC_C[angle].fDetV.x;
	const float fDetVY = gC_C[angle].fDetV.y;
	const float fDetVZ = gC_C[angle].fDetV.z;
	const float fCylCX = gC_C[angle].fCylC.x;
	const float fCylCY = gC_C[angle].fCylC.y;
	const float fCylCZ = gC_C[angle].fCylC.z;

	// There's a bit of redundancy in these if we need to save space:
	const float fCylAX = gC_C[angle].fCylA.x;
	const float fCylAY = gC_C[angle].fCylA.y;
	const float fCylAZ = gC_C[angle].fCylA.z;
	const float fCylBX = gC_C[angle].fCylB.x;
	const float fCylBY = gC_C[angle].fCylB.y;
	const float fCylBZ = gC_C[angle].fCylB.z;
	const float fDetUT = gC_C[angle].fDetUT;

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

	const float fDetU = (2*detectorU - (int)dims.iProjU + 1) * 0.5f * fDetUT;
	float fsindu, fcosdu;
	sincosf(fDetU, &fsindu, &fcosdu);

	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)
	{
		const float fDetV = detectorV - 0.5f * dims.iProjV + 0.5f;
		/* Trace ray from Src to (detectorU,detectorV) from */
		/* X = startSlice to X = endSlice                   */

		const float fDetX = fCylCX + fCylAX*fcosdu + fCylBX*fsindu + fDetVX*fDetV;
		const float fDetY = fCylCY + fCylAY*fcosdu + fCylBY*fsindu + fDetVY*fDetV;
		const float fDetZ = fCylCZ + fCylAZ*fcosdu + fCylBZ*fsindu + fDetVZ*fDetV;

		//printf("%d %d -> %f %f %f\n", detectorU, detectorV, fDetX, fDetY, fDetZ);

		/*        (x)   ( 1)       ( 0) */
		/* ray:   (y) = (ay) * x + (by) */
		/*        (z)   (az)       (bz) */
		

		float a1 = (c.c1(fSrcX,fSrcY,fSrcZ) - c.c1(fDetX,fDetY,fDetZ)) / (c.c0(fSrcX,fSrcY,fSrcZ) - c.c0(fDetX,fDetY,fDetZ)) ;

		float a2 = (c.c2(fSrcX,fSrcY,fSrcZ) - c.c2(fDetX,fDetY,fDetZ)) / (c.c0(fSrcX,fSrcY,fSrcZ) - c.c0(fDetX,fDetY,fDetZ)) ;

		const float fDistCorr = sqrt(a1*a1 + a2*a2 + 1.0f) * c.c0(fvx, fvy, fvz);

		// a1, a2 are in global units above. Converted to texture steps below.

		a1 /= c.c1(fvx, fvy, fvz);
		a2 /= c.c2(fvx, fvy, fvz);

		const float b1 = c.c1(fSrcX,fSrcY,fSrcZ) / c.c1(fvx, fvy, fvz) - a1 * c.c0(fSrcX,fSrcY,fSrcZ);
		const float b2 = c.c2(fSrcX,fSrcY,fSrcZ) / c.c2(fvx, fvy, fvz) - a2 * c.c0(fSrcX,fSrcY,fSrcZ);

		a1 *= c.c0(fvx, fvy, fvz);
		a2 *= c.c0(fvx, fvy, fvz);


		//const float fVal = D_projData[(detectorV*dims.iProjAngles+angle)*projPitch+detectorU]  * fDistCorr;

		const float fVal = tex3D<float>(tex, detectorU + 0.5f, relAngle + angle + 0.5f, detectorV + 0.5f) * fDistCorr;

		//float f0 = startSlice + 0.5f;

		// f1 == 2.0 for middle of first row
		float f1 = a1 * (startSlice + (0.5 - 0.5f*c.nSlices(dims)) ) + b1 + 0.5f*c.nDim1(dims) + 1.5f; // NB: +1.5
		float f2 = a2 * (startSlice + (0.5 - 0.5f*c.nSlices(dims)) ) + b2 + 0.5f*c.nDim2(dims) + 1.5f; // NB: +1.5

		for (int s = startSlice; s < endSlice; ++s)
		{

			// Can use modff to get integer and fractional results.
			// but modff rounds towards zero, so need to be careful there
			// (All actually used indices are non-negative, so only
			//  potential issue on the border.)

#if 0
			float if1, if2;
			float z1 = modff(f1, &if1);
			float z2 = modff(f2, &if2);

			int c1 = (int)floorf(if1);
			int c2 = (int)floorf(if2);
#else
			int c1 = (int)floorf(f1);
			int c2 = (int)floorf(f2);
			float z1 = f1 - c1;
			float z2 = f2 - c2;
#endif

			// TODO: Scale the interpolation kernel to handle anisotropic voxels
			//z1 = __saturatef(c.c1(fvx, fvy, fvz) * (z1 - 0.5f) + 0.5f);
			//z2 = __saturatef(c.c2(fvx, fvy, fvz) * (z2 - 0.5f) + 0.5f);


			float* addr;
			// TODO: ensure this works for large volumes (int vs size_t)
			// TODO: why are these two not equivalent? (implicit type promotion?)
			//addr = D_volData + (c.iz(s,c1-2,c2-2)*dims.iVolY +c.iy(s,c1-2,c2-2))*volPitch + c.ix(s,c1-2,c2-2);
			addr = D_volData + (c.iz(s,c1,c2)*dims.iVolY +c.iy(s,c1,c2))*volPitch + c.ix(s,c1,c2) - 2 * c.stride1(dims, volPitch) - 2 * c.stride2(dims, volPitch);
#if 0
			int d1 =  (c.iz(s,c1-2,c2-2)*dims.iVolY +c.iy(s,c1-2,c2-2))*volPitch + c.ix(s,c1-2,c2-2);
			int d2 = (c.iz(s,c1,c2)*dims.iVolY +c.iy(s,c1,c2))*volPitch + c.ix(s,c1,c2) - 2 * c.stride1(dims, volPitch) - 2 * c.stride2(dims, volPitch);
			assert(d1 == d2);
			float *fd1 = D_volData + ((c.iz(s,c1-2,c2-2)*dims.iVolY +c.iy(s,c1-2,c2-2))*volPitch + c.ix(s,c1-2,c2-2));
			float *fd2 = D_volData + (c.iz(s,c1,c2)*dims.iVolY +c.iy(s,c1,c2))*volPitch + c.ix(s,c1,c2) - 2 * c.stride1(dims, volPitch) - 2 * c.stride2(dims, volPitch);
			assert(fd1 == fd2);
#endif

			// TODO: handle borders better to skip checks in the inner loop

			if (c1 > 1 && c1-1 <= c.nDim1(dims)) {
				if (c2 > 1 && c2-1 <= c.nDim2(dims)) {
					atomicAdd(addr, fVal * (1.0f - z1) * (1.0f - z2));
				}
				if (c2 > 0 && c2-1 < c.nDim2(dims)) {
					atomicAdd(addr + c.stride2(dims, volPitch), fVal * (1.0f - z1) * z2);
				}
			}
			if (c1 > 0 && c1-1 < c.nDim1(dims)) {
				if (c2 > 1 && c2-1 <= c.nDim2(dims)) {
					atomicAdd(addr + c.stride1(dims, volPitch), fVal * z1 * (1.0f - z2));
				}
				if (c2 > 0 && c2-1 < c.nDim2(dims)) {
					atomicAdd(addr + c.stride1(dims, volPitch) + c.stride2(dims, volPitch), fVal * z1 * z2);
				}
			}

			//fVal += c.tex(f0, f1, f2);
			//f0 += 1.0f;
//printf("B: %d %d,%d %d,%f (%d),%f (%d) %f,%f\n", angle, detectorV, detectorU, s, f1, c1, f2, c2, z1, z2);
			f1 += a1;
			f2 += a2;
		}

		//D_projData[(detectorV*dims.iProjAngles+angle)*projPitch+detectorU] += fVal;
	}
}



bool ConeCylFP_Array_internal(cudaPitchedPtr D_projData,
                  cudaTextureObject_t D_texObj,
                  const SDimensions3D& dims, unsigned int angleCount, const SCylConeProjection* angles,
                  const SProjectorParams3D& params)
{

	transferConstants(angles, angleCount, params);

	std::list<cudaStream_t> streams;
	dim3 dimBlock(g_detBlockU, g_anglesPerBlock); // region size, angles

	// Run over all angles, grouping them into groups of the same
	// orientation (roughly horizontal vs. roughly vertical).
	// Start a stream of grids for each such group.

	unsigned int blockStart = 0;
	unsigned int blockEnd = 0;
	int blockDirection = 0;

	assert(angleCount > 0);
	float fDetRadius = angles[0].fDetR;

	// timeval t;
	// tic(t);

	for (unsigned int a = 0; a <= angleCount; ++a) {
		int dir = -1;
		if (a != angleCount) {
			float dX = fabsf(angles[a].fSrcX - angles[a].fDetCX);
			float dY = fabsf(angles[a].fSrcY - angles[a].fDetCY);
			float dZ = fabsf(angles[a].fSrcZ - angles[a].fDetCZ);

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

				cudaStream_t stream;
				cudaStreamCreate(&stream);
				streams.push_back(stream);

				// printf("angle block: %d to %d, %d (%dx%d, %dx%d)\n", blockStart, blockEnd, blockDirection, dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

				if (blockDirection == 0) {
					for (unsigned int i = 0; i < dims.iVolX; i += g_blockSlices)
						cylcone_FP_t<DIR_X><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), D_texObj, i, blockStart, blockEnd, dims, fDetRadius, params.fVolScaleX, params.fVolScaleY, params.fVolScaleZ, params.fOutputScale);
				} else if (blockDirection == 1) {
					for (unsigned int i = 0; i < dims.iVolY; i += g_blockSlices)
						cylcone_FP_t<DIR_Y><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), D_texObj, i, blockStart, blockEnd, dims, fDetRadius, params.fVolScaleX, params.fVolScaleY, params.fVolScaleZ, params.fOutputScale);
				} else if (blockDirection == 2) {
					for (unsigned int i = 0; i < dims.iVolZ; i += g_blockSlices)
						cylcone_FP_t<DIR_Z><<<dimGrid, dimBlock, 0, stream>>>((float*)D_projData.ptr, D_projData.pitch/sizeof(float), D_texObj, i, blockStart, blockEnd, dims, fDetRadius, params.fVolScaleX, params.fVolScaleY, params.fVolScaleZ, params.fOutputScale);
				}

			}

			blockDirection = dir;
			blockStart = a;
		}
	}

	bool ok = true;

	for (std::list<cudaStream_t>::iterator iter = streams.begin(); iter != streams.end(); ++iter) {
		ok &= checkCuda(cudaStreamSynchronize(*iter), "cone curved fp");
		cudaStreamDestroy(*iter);
	}

	// printf("%f\n", toc(t));

	return ok;
}

bool ConeCylBP_Array_internal(cudaPitchedPtr D_volData,
                  cudaTextureObject_t D_texObj,
                  const SDimensions3D& dims, unsigned int startAngle, unsigned int angleCount, const SCylConeProjection* angles,
                  const SProjectorParams3D& params)
{
	angles += startAngle;

	transferConstants(angles, angleCount, params);

	std::list<cudaStream_t> streams;
	dim3 dimBlock(g_detBlockU, g_anglesPerBlock); // region size, angles

	// Run over all angles, grouping them into groups of the same
	// orientation (roughly horizontal vs. roughly vertical).
	// Start a stream of grids for each such group.

	unsigned int blockStart = 0;
	unsigned int blockEnd = 0;
	int blockDirection = 0;

	assert(angleCount > 0);
	float fDetRadius = angles[0].fDetR;

	// timeval t;
	// tic(t);

	for (unsigned int a = 0; a <= angleCount; ++a) {
		int dir = -1;
		if (a != angleCount) {
			float dX = fabsf(angles[a].fSrcX - angles[a].fDetCX);
			float dY = fabsf(angles[a].fSrcY - angles[a].fDetCY);
			float dZ = fabsf(angles[a].fSrcZ - angles[a].fDetCZ);

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

				cudaStream_t stream;
				cudaStreamCreate(&stream);
				streams.push_back(stream);

				// printf("angle block: %d to %d, %d (%dx%d, %dx%d)\n", blockStart, blockEnd, blockDirection, dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

				if (blockDirection == 0) {
					for (unsigned int i = 0; i < dims.iVolX; i += g_blockSlices)
						cylcone_BP_t<DIR_X><<<dimGrid, dimBlock, 0, stream>>>((float*)D_volData.ptr, D_volData.pitch/sizeof(float), D_texObj, i, startAngle, blockStart, blockEnd, dims, fDetRadius, params.fVolScaleX, params.fVolScaleY, params.fVolScaleZ);
				} else if (blockDirection == 1) {
					for (unsigned int i = 0; i < dims.iVolY; i += g_blockSlices)
						cylcone_BP_t<DIR_Y><<<dimGrid, dimBlock, 0, stream>>>((float*)D_volData.ptr, D_volData.pitch/sizeof(float), D_texObj, i, startAngle, blockStart, blockEnd, dims, fDetRadius, params.fVolScaleX, params.fVolScaleY, params.fVolScaleZ);
				} else if (blockDirection == 2) {
					for (unsigned int i = 0; i < dims.iVolZ; i += g_blockSlices)
						cylcone_BP_t<DIR_Z><<<dimGrid, dimBlock, 0, stream>>>((float*)D_volData.ptr, D_volData.pitch/sizeof(float), D_texObj, i, startAngle, blockStart, blockEnd, dims, fDetRadius, params.fVolScaleX, params.fVolScaleY, params.fVolScaleZ);
				}

			}

			blockDirection = dir;
			blockStart = a;
		}
	}

	bool ok = true;

	for (std::list<cudaStream_t>::iterator iter = streams.begin(); iter != streams.end(); ++iter) {
		ok &= checkCuda(cudaStreamSynchronize(*iter), "cone cyl bp");
		cudaStreamDestroy(*iter);
	}

	// printf("%f\n", toc(t));

	return ok;
}


bool ConeCylFP(cudaPitchedPtr D_volumeData,
            cudaPitchedPtr D_projData,
            const SDimensions3D& dims, const SCylConeProjection* angles,
            const SProjectorParams3D& params)
{
	// transfer volume to array

	cudaArray* cuArray = allocateVolumeArray(dims);
	transferVolumeToArray(D_volumeData, cuArray, dims);

	cudaTextureObject_t D_texObj;
	if (!createTextureObject3D(cuArray, D_texObj)) {
		cudaFreeArray(cuArray);
		return false;
	}

	bool ret;

	for (unsigned int iAngle = 0; iAngle < dims.iProjAngles; iAngle += g_MaxAngles) {
		unsigned int iEndAngle = iAngle + g_MaxAngles;
		if (iEndAngle >= dims.iProjAngles)
			iEndAngle = dims.iProjAngles;

		cudaPitchedPtr D_subprojData = D_projData;
		D_subprojData.ptr = (char*)D_projData.ptr + iAngle * D_projData.pitch;

		ret = ConeCylFP_Array_internal(D_subprojData, D_texObj,
		                            dims, iEndAngle - iAngle, angles + iAngle,
		                            params);
		if (!ret)
			break;
	}

	cudaFreeArray(cuArray);

	return ret;
}

bool ConeCylBP_Array_matched(cudaPitchedPtr D_volumeData,
                  cudaArray *D_projArray,
                  const SDimensions3D& dims, const SCylConeProjection* angles,
                  const SProjectorParams3D& params)
{
	cudaTextureObject_t D_texObj;
	if (!createTextureObject3D(D_projArray, D_texObj))
		return false;

	bool ret = true;

	for (unsigned int iAngle = 0; iAngle < dims.iProjAngles; iAngle += g_MaxAngles) {
		unsigned int iEndAngle = iAngle + g_MaxAngles;
		if (iEndAngle >= dims.iProjAngles)
			iEndAngle = dims.iProjAngles;

		ret = ConeCylBP_Array_internal(D_volumeData, D_texObj, dims, iAngle, iEndAngle - iAngle, angles, params);
		if (!ret)
			break;
	}

	cudaDestroyTextureObject(D_texObj);

	return ret;
}

bool ConeCylBP_matched(cudaPitchedPtr D_volumeData,
            cudaPitchedPtr D_projData,
            const SDimensions3D& dims, const SCylConeProjection* angles,
            const SProjectorParams3D& params)
{
	// transfer projections to array

	cudaArray* cuArray = allocateProjectionArray(dims);
	transferProjectionsToArray(D_projData, cuArray, dims);

	bool ret = ConeCylBP_Array_matched(D_volumeData, cuArray, dims, angles, params);

	cudaFreeArray(cuArray);

	return ret;
}



}
