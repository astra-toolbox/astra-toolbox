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

#include "astra/cuda/gpu_runtime_wrapper.h"

#include <cstdio>
#include <cassert>

#include "astra/cuda/3d/util3d.h"

#ifdef STANDALONE
#include "testutil.h"
#endif

#include "astra/cuda/3d/dims3d.h"
#include "astra/cuda/3d/proj3d.h"


static const unsigned int g_anglesPerBlock = 8;
// TODO: clean this up
// NB: This is smaller than g_cuda3d_blockSlices, which means CGM will
// overestimate how much memory we need, which is not ideal, but also not a
// major problem.
static const unsigned int g_blockSlices = 24;

static const unsigned int g_detBlockU = 32;
static const unsigned int g_detBlockV = 16;

static const unsigned g_MaxAngles = 960;


#include "ray3d.inl"


namespace astraCUDA3d {

struct DevCylConeParams {
	float3 fSrc;
	float3 fCylC;
	float3 fCylA;
	float3 fCylB;
	float3 fDetV;
	float fDetUT;
};

__constant__ DevCylConeParams gC_C[g_MaxAngles];
__constant__ int gC_index[g_MaxAngles];


using TransferConstantsBuffer = TransferConstantsBuffer_t<DevCylConeParams>;

class ConeCylFP_t : public RayDrivenProjector3D<TransferConstantsBuffer, SCylConeProjection> {
public:
	ConeCylFP_t(float fDetRadius) : m_fDetRadius(fDetRadius) { }

	virtual bool sortProjections(const SDimensions3D& dims,
	                             const std::vector<SCylConeProjection> &angles,
	                             std::vector<SCylConeProjection>& anglesX,
	                             std::vector<SCylConeProjection>& anglesY,
	                             std::vector<SCylConeProjection>& anglesZ,
	                             std::vector<int>& indicesX,
	                             std::vector<int>& indicesY,
	                             std::vector<int>& indicesZ) override;
	virtual bool transferConstants(const SCylConeProjection* angles, const int* indices,
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
	virtual bool dispatchKernel_BP(float *D_volData, unsigned int volPitch,
	                            const float *D_projData, unsigned int projPitch,
	                            float fStartSlice,
	                            const SDimensions3D& dims,
	                            const SDimensions3D& blockDims,
	                            const SProjectorParams3D& params,
	                            int majordim,
	                            cudaStream_t stream) override;

private:
	float m_fDetRadius; // HACK
};


bool ConeCylFP_t::transferConstants(const SCylConeProjection* angles, const int* indices, unsigned int iProjAngles,
                                    TransferConstantsBuffer& buf, cudaStream_t stream)
{
	DevCylConeParams *p = &(std::get<0>(buf.d))[0];

	// We use an event to assure that the previous transferConstants has completed before
	// re-using the buffer. (Even if it is very unlikely that it hasn't.)
	bool ok = checkCuda(cudaStreamWaitEvent(stream, buf.event, 0), "transferConstants wait");

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

	ok &= checkCuda(cudaMemcpyToSymbolAsync(gC_C, p, iProjAngles*sizeof(DevCylConeParams), 0, cudaMemcpyHostToDevice, stream), "transferConstants transfer");

	ok &= checkCuda(cudaMemcpyToSymbolAsync(gC_index, indices, iProjAngles * sizeof(int), 0, cudaMemcpyHostToDevice, stream), "transferConstants transfer indices");

	ok &= checkCuda(cudaEventRecord(buf.event, stream), "transferConstants event");

	return ok;
}


__global__ void cylcone_FP_t(float* D_projData, unsigned int projPitch,
                          cudaTextureObject_t tex,
                          float fStartSlice,
                          const SDimensions3D blockDims, unsigned int totalAngles,
                          float fR, // cylinder radius
                          float fvx, float fvy, float fvz, // voxel sizes
                          float fOutputScale
                         )
{
	int angle = blockIdx.z * blockDim.y + threadIdx.y;
	if (angle >= blockDims.iProjAngles)
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

	// Get original index of this projection (for in the output array)
	angle = gC_index[angle];

	const int detectorU = blockIdx.x * blockDim.x + threadIdx.x;
	if (detectorU >= blockDims.iProjU)
		return;
	const int startDetectorV = blockIdx.y * g_detBlockV;
	int endDetectorV = startDetectorV + g_detBlockV;
	if (endDetectorV > blockDims.iProjV)
		endDetectorV = blockDims.iProjV;

	int sliceCount = g_blockSlices;
	if (sliceCount > blockDims.iVolZ)
		sliceCount = blockDims.iVolZ;

	const float fDetU = (2*detectorU - (int)blockDims.iProjU + 1) * 0.5f * fDetUT;
	float fsindu, fcosdu;
	sincosf(fDetU, &fsindu, &fcosdu);

	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)
	{
		const float fDetV = detectorV - 0.5f * blockDims.iProjV + 0.5f;
		/* Trace ray from Src to (detectorU,detectorV) from */
		/* Z = 0 to Z = sliceCount                   */

		const float fDetX = fCylCX + fCylAX*fcosdu + fCylBX*fsindu + fDetVX*fDetV;
		const float fDetY = fCylCY + fCylAY*fcosdu + fCylBY*fsindu + fDetVY*fDetV;
		const float fDetZ = fCylCZ + fCylAZ*fcosdu + fCylBZ*fsindu + fDetVZ*fDetV;

		//printf("%d %d -> %f %f %f\n", detectorU, detectorV, fDetX, fDetY, fDetZ);

		/*        (x)   (ax)       (bx)    */
		/* ray:   (y) = (ay) * z + (by)    */
		/*        (z)   ( 1)       ( 0)    */
		

		float ax = (fSrcX - fDetX) / (fSrcZ - fDetZ);
		float ay = (fSrcY - fDetY) / (fSrcZ - fDetZ);

		const float fDistCorr = sqrt(ax*ax + ay*ay + 1.0f) * fvz;

		// a1, a2 are in global units above. Converted to texture steps below.

		ax /= fvx;
		ay /= fvy;

		const float bx = fSrcX / fvx - ax * fSrcZ;
		const float by = fSrcY / fvy - ay * fSrcZ;

		ax *= fvz;
		ay *= fvz;


		float fVal = 0.0f;

		float fx = ax * fStartSlice + bx + 0.5f*blockDims.iVolX;
		float fy = ay * fStartSlice + by + 0.5f*blockDims.iVolY;

		for (int s = 0; s < sliceCount; ++s)
		{
//printf("F: %d %d,%d %f,%f,%f\n", angle, detectorV, detectorU, f0, f1, f2);
			// TODO: Move away from texture interpolation for (strongly) anisotropic pixels
			// See BP for coefficients
			fVal += tex2DLayered<float>(tex, fx, fy, s);
			fx += ax;
			fy += ay;
		}

		fVal *= fDistCorr * fOutputScale;

		D_projData[(detectorV*totalAngles+angle)*projPitch+detectorU] += fVal;
	}
}

__global__ void cylcone_BP_t(float* D_volData, unsigned int volPitch,
                          const float* D_projData, unsigned int projPitch,
                          float fStartSlice,
                          const SDimensions3D blockDims, unsigned int totalAngles,
                          float fR, // cylinder radius
                          float fvx, float fvy, float fvz // voxel sizes
                         )
{
	int angle = blockIdx.z * blockDim.y + threadIdx.y;
	if (angle >= blockDims.iProjAngles)
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

	// Get original index of this projection (for in the output array)
	angle = gC_index[angle];

	const int detectorU = blockIdx.x * blockDim.x + threadIdx.x;
	if (detectorU >= blockDims.iProjU)
		return;
	const int startDetectorV = blockIdx.y * g_detBlockV;
	int endDetectorV = startDetectorV + g_detBlockV;
	if (endDetectorV > blockDims.iProjV)
		endDetectorV = blockDims.iProjV;

	int sliceCount = g_blockSlices;
	if (sliceCount > blockDims.iVolZ)
		sliceCount = blockDims.iVolZ;

	const float fDetU = (2*detectorU - (int)blockDims.iProjU + 1) * 0.5f * fDetUT;
	float fsindu, fcosdu;
	sincosf(fDetU, &fsindu, &fcosdu);

	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)
	{
		const float fDetV = detectorV - 0.5f * blockDims.iProjV + 0.5f;
		/* Trace ray from Src to (detectorU,detectorV) from */
		/* Z = 0 to Z = sliceCount                   */

		const float fDetX = fCylCX + fCylAX*fcosdu + fCylBX*fsindu + fDetVX*fDetV;
		const float fDetY = fCylCY + fCylAY*fcosdu + fCylBY*fsindu + fDetVY*fDetV;
		const float fDetZ = fCylCZ + fCylAZ*fcosdu + fCylBZ*fsindu + fDetVZ*fDetV;

		//printf("%d %d -> %f %f %f\n", detectorU, detectorV, fDetX, fDetY, fDetZ);

		/*        (x)   (ax)       (bx)    */
		/* ray:   (y) = (ay) * z + (by)    */
		/*        (z)   ( 1)       ( 0)    */
		
		float ax = (fSrcX - fDetX) / (fSrcZ - fDetZ);
		float ay = (fSrcY - fDetY) / (fSrcZ - fDetZ);

		const float fDistCorr = sqrt(ax*ax + ay*ay + 1.0f) * fvz;

		// a1, a2 are in global units above. Converted to texture steps below.

		ax /= fvx;
		ay /= fvy;

		const float bx = fSrcX / fvx - ax * fSrcZ;
		const float by = fSrcY / fvy - ax * fSrcZ;

		ax *= fvz;
		ay *= fvz;

		const float fVal = D_projData[(size_t)(detectorV*totalAngles+angle)*projPitch+detectorU] * fDistCorr;
		//const float fVal = tex3D<float>(tex, detectorU + 0.5f, relAngle + angle + 0.5f, detectorV + 0.5f) * fDistCorr;

		//float f0 = startSlice + 0.5f;

		// fx == 2.0 for middle of first row
		float fx = ax * fStartSlice + bx + 0.5f*blockDims.iVolX + 1.5f; // NB: +1.5
		float fy = ay * fStartSlice + by + 0.5f*blockDims.iVolY + 1.5f; // NB: +1.5

		for (int s = 0; s < sliceCount; ++s)
		{

			// Can use modff to get integer and fractional results.
			// but modff rounds towards zero, so need to be careful there
			// (All actually used indices are non-negative, so only
			//  potential issue on the border.)

#if 0
			float ifx, ify;
			float zx = modff(fx, &ifx);
			float zy = modff(fy, &ify);

			int cx = (int)floorf(ifx);
			int cy = (int)floorf(ify);
#else
			int cx = (int)floorf(fx);
			int cy = (int)floorf(fy);
			float zx = fx - cx;
			float zy = fy - cy;
#endif

			// TODO: Scale the interpolation kernel to handle anisotropic voxels
			//zx = __saturatef(fvx * (zx - 0.5f) + 0.5f);
			//zy = __saturatef(fvy * (zy - 0.5f) + 0.5f);


			// TODO: ensure this works for large volumes (int vs size_t)
			// TODO: why are these two not equivalent? (implicit type promotion?)
			//addr = D_volData + (c.iz(s,c1-2,c2-2)*dims.iVolY +c.iy(s,c1-2,c2-2))*volPitch + c.ix(s,c1-2,c2-2);
			float * const addr = D_volData + (size_t)(s*blockDims.iVolY + cy)*volPitch + cx - 2 * volPitch - 2;
#if 0
			int d1 =  (c.iz(s,c1-2,c2-2)*dims.iVolY +c.iy(s,c1-2,c2-2))*volPitch + c.ix(s,c1-2,c2-2);
			int d2 = (c.iz(s,c1,c2)*dims.iVolY +c.iy(s,c1,c2))*volPitch + c.ix(s,c1,c2) - 2 * c.stride1(dims, volPitch) - 2 * c.stride2(dims, volPitch);
			assert(d1 == d2);
			float *fd1 = D_volData + ((c.iz(s,c1-2,c2-2)*dims.iVolY +c.iy(s,c1-2,c2-2))*volPitch + c.ix(s,c1-2,c2-2));
			float *fd2 = D_volData + (c.iz(s,c1,c2)*dims.iVolY +c.iy(s,c1,c2))*volPitch + c.ix(s,c1,c2) - 2 * c.stride1(dims, volPitch) - 2 * c.stride2(dims, volPitch);
			assert(fd1 == fd2);
#endif

			// TODO: handle borders better to skip checks in the inner loop

			if (cx > 1 && cx-1 <= blockDims.iVolX) {
				if (cy > 1 && cy-1 <= blockDims.iVolY) {
					atomicAdd(addr, fVal * (1.0f - zx) * (1.0f - zy));
				}
				if (cy > 0 && cy-1 < blockDims.iVolY) {
					atomicAdd(addr + volPitch, fVal * (1.0f - zx) * zy);
				}
			}
			if (cx > 0 && cx-1 < blockDims.iVolX) {
				if (cy > 1 && cy-1 <= blockDims.iVolY) {
					atomicAdd(addr + 1, fVal * zx * (1.0f - zy));
				}
				if (cy > 0 && cy-1 < blockDims.iVolY) {
					atomicAdd(addr + 1 + volPitch, fVal * zx * zy);
				}
			}

			//fVal += c.tex(f0, f1, f2);
			//f0 += 1.0f;
//printf("B: %d %d,%d %d,%f (%d),%f (%d) %f,%f\n", angle, detectorV, detectorU, s, f1, c1, f2, c2, z1, z2);
			fx += ax;
			fy += ay;
		}

		//D_projData[(detectorV*dims.iProjAngles+angle)*projPitch+detectorU] += fVal;
	}
}


bool ConeCylFP_t::sortProjections(const SDimensions3D& dims, const std::vector<SCylConeProjection>& angles,
                               std::vector<SCylConeProjection>& anglesX,
                               std::vector<SCylConeProjection>& anglesY,
                               std::vector<SCylConeProjection>& anglesZ,
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
		float dX = fabsf(angles[i].fSrcX - angles[i].fDetCX);
		float dY = fabsf(angles[i].fSrcY - angles[i].fDetCY);
		float dZ = fabsf(angles[i].fSrcZ - angles[i].fDetCZ);

		if (dX >= dY && dX >= dZ) {
			// X
			// X <- Y, Y <- Z, Z <- X
			SCylConeProjection a;
			a.fSrcX = angles[i].fSrcY;
			a.fSrcY = angles[i].fSrcZ;
			a.fSrcZ = angles[i].fSrcX;
			a.fDetCX = angles[i].fDetCY;
			a.fDetCY = angles[i].fDetCZ;
			a.fDetCZ = angles[i].fDetCX;
			a.fDetUX = angles[i].fDetUY;
			a.fDetUY = angles[i].fDetUZ;
			a.fDetUZ = angles[i].fDetUX;
			a.fDetVX = angles[i].fDetVY;
			a.fDetVY = angles[i].fDetVZ;
			a.fDetVZ = angles[i].fDetVX;
			a.fDetR = angles[i].fDetR;
			//fprintf(stderr, "Index %d -> X\n", i);
			anglesX.push_back(a);
			indicesX.push_back(i);
		} else if (dY >= dX && dY >= dZ) {
			// Y
			// X <- X, Y <- Z, Z <- Y
			SCylConeProjection a;
			a.fSrcX = angles[i].fSrcX;
			a.fSrcY = angles[i].fSrcZ;
			a.fSrcZ = angles[i].fSrcY;
			a.fDetCX = angles[i].fDetCX;
			a.fDetCY = angles[i].fDetCZ;
			a.fDetCZ = angles[i].fDetCY;
			a.fDetUX = angles[i].fDetUX;
			a.fDetUY = angles[i].fDetUZ;
			a.fDetUZ = angles[i].fDetUY;
			a.fDetVX = angles[i].fDetVX;
			a.fDetVY = angles[i].fDetVZ;
			a.fDetVZ = angles[i].fDetVY;
			a.fDetR = angles[i].fDetR;
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

bool ConeCylFP_t::dispatchKernel(float *D_projData, unsigned int projPitch,
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
	dim3 dimGrid = computeGrid3Ceil(blockDims.iProjU, g_detBlockU, blockDims.iProjV, g_detBlockV, blockDims.iProjAngles, g_anglesPerBlock);

	cylcone_FP_t<<<dimGrid, dimBlock, 0, stream>>>(D_projData, projPitch, D_texObj, fStartSlice, blockDims, dims.iProjAngles, m_fDetRadius, params.volScale.fX, params.volScale.fY, params.volScale.fZ, params.fOutputScale);

	return true;
}

bool ConeCylFP_t::dispatchKernel_BP(float *D_volData, unsigned int volPitch,
                                    const float *D_projData, unsigned int projPitch,
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
	dim3 dimGrid = computeGrid3Ceil(blockDims.iProjU, g_detBlockU, blockDims.iProjV, g_detBlockV, blockDims.iProjAngles, g_anglesPerBlock);

	cylcone_BP_t<<<dimGrid, dimBlock, 0, stream>>>(D_volData, volPitch, D_projData, projPitch, fStartSlice, blockDims, dims.iProjAngles, m_fDetRadius, params.volScale.fX, params.volScale.fY, params.volScale.fZ); // TODO? params.fOutputScale);

	return true;
}



bool ConeCylFP(cudaPitchedPtr D_volumeData,
            cudaPitchedPtr D_projData,
            const SDimensions3D& dims, const std::vector<SCylConeProjection>& angles,
            const SProjectorParams3D& params)
{
	if (angles.empty())
		return false;

	ConeCylFP_t CC(angles[0].fDetR);
	return CC.project(D_volumeData, D_projData, ConeCylFP_t::MODE_FORWARD, dims, angles, params);
}

bool ConeCylBP_matched(cudaPitchedPtr D_volumeData,
            cudaPitchedPtr D_projData,
            const SDimensions3D& dims, const std::vector<SCylConeProjection>& angles,
            const SProjectorParams3D& params)
{
	if (angles.empty())
		return false;

	ConeCylFP_t CC(angles[0].fDetR);
	return CC.project(D_volumeData, D_projData, ConeCylFP_t::MODE_BACKWARD, dims, angles, params);
}

}
