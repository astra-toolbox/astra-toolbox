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

#include "astra/cuda/2d/util.h"
#include "astra/cuda/2d/arith.h"

#include <cstdio>
#include <cassert>
#include <iostream>


typedef texture<float, 2, cudaReadModeElementType> texture2D;

static texture2D gT_FanProjTexture;


namespace astraCUDA {

const unsigned int g_anglesPerBlock = 16;
const unsigned int g_blockSliceSize = 32;
const unsigned int g_blockSlices = 16;

const unsigned int g_MaxAngles = 2560;

struct DevFanParams {
	float fNumC;
	float fNumX;
	float fNumY;
	float fDenC;
	float fDenX;
	float fDenY;
};

__constant__ DevFanParams gC_C[g_MaxAngles];


static bool bindProjDataTexture(float* data, unsigned int pitch, unsigned int width, unsigned int height, cudaTextureAddressMode mode = cudaAddressModeBorder)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	gT_FanProjTexture.addressMode[0] = mode;
	gT_FanProjTexture.addressMode[1] = mode;
	gT_FanProjTexture.filterMode = cudaFilterModeLinear;
	gT_FanProjTexture.normalized = false;

	cudaBindTexture2D(0, gT_FanProjTexture, (const void*)data, channelDesc, width, height, sizeof(float)*pitch);

	// TODO: error value?

	return true;
}

template<bool FBPWEIGHT>
__global__ void devFanBP(float* D_volData, unsigned int volPitch, unsigned int startAngle, const SDimensions dims, float fOutputScale)
{
	const int relX = threadIdx.x;
	const int relY = threadIdx.y;

	int endAngle = startAngle + g_anglesPerBlock;
	if (endAngle > dims.iProjAngles)
		endAngle = dims.iProjAngles;
	const int X = blockIdx.x * g_blockSlices + relX;
	const int Y = blockIdx.y * g_blockSliceSize + relY;

	if (X >= dims.iVolWidth || Y >= dims.iVolHeight)
		return;

	const float fX = ( X - 0.5f*dims.iVolWidth + 0.5f );
	const float fY = - ( Y - 0.5f*dims.iVolHeight + 0.5f );

	float* volData = (float*)D_volData;

	float fVal = 0.0f;
	float fA = startAngle + 0.5f;

	for (int angle = startAngle; angle < endAngle; ++angle)
	{
		const float fNumC = gC_C[angle].fNumC;
		const float fNumX = gC_C[angle].fNumX;
		const float fNumY = gC_C[angle].fNumY;
		const float fDenX = gC_C[angle].fDenX;
		const float fDenY = gC_C[angle].fDenY;

		const float fNum = fNumC + fNumX * fX + fNumY * fY;
		const float fDen = (FBPWEIGHT ? 1.0 : gC_C[angle].fDenC) + fDenX * fX + fDenY * fY;

		// Scale factor is the approximate number of rays traversing this pixel,
		// given by the inverse size of a detector pixel scaled by the magnification
		// factor of this pixel.
		// Magnification factor is || u (d-s) || / || u (x-s) ||

		const float fr = __fdividef(1.0f, fDen);
		const float fT = fNum * fr;
		fVal += tex2D(gT_FanProjTexture, fT, fA) * (FBPWEIGHT ? fr * fr : fr);
		fA += 1.0f;
	}

	volData[Y*volPitch+X] += fVal * fOutputScale;
}

// supersampling version
__global__ void devFanBP_SS(float* D_volData, unsigned int volPitch, unsigned int startAngle, const SDimensions dims, float fOutputScale)
{
	const int relX = threadIdx.x;
	const int relY = threadIdx.y;

	int endAngle = startAngle + g_anglesPerBlock;
	if (endAngle > dims.iProjAngles)
		endAngle = dims.iProjAngles;
	const int X = blockIdx.x * g_blockSlices + relX;
	const int Y = blockIdx.y * g_blockSliceSize + relY;

	if (X >= dims.iVolWidth || Y >= dims.iVolHeight)
		return;

	const float fXb = ( X - 0.5f*dims.iVolWidth + 0.5f - 0.5f + 0.5f/dims.iRaysPerPixelDim);
	const float fYb = - ( Y - 0.5f*dims.iVolHeight + 0.5f - 0.5f + 0.5f/dims.iRaysPerPixelDim);

	const float fSubStep = 1.0f/dims.iRaysPerPixelDim;

	float* volData = (float*)D_volData;

	fOutputScale /= (dims.iRaysPerPixelDim * dims.iRaysPerPixelDim);

	float fVal = 0.0f;
	float fA = startAngle + 0.5f;

	for (int angle = startAngle; angle < endAngle; ++angle)
	{
		const float fNumC = gC_C[angle].fNumC;
		const float fNumX = gC_C[angle].fNumX;
		const float fNumY = gC_C[angle].fNumY;
		const float fDenC = gC_C[angle].fDenC;
		const float fDenX = gC_C[angle].fDenX;
		const float fDenY = gC_C[angle].fDenY;

		// TODO: Optimize these loops...
		float fX = fXb;
		for (int iSubX = 0; iSubX < dims.iRaysPerPixelDim; ++iSubX) {
			float fY = fYb;
			for (int iSubY = 0; iSubY < dims.iRaysPerPixelDim; ++iSubY) {

				const float fNum = fNumC + fNumX * fX + fNumY * fY;
				const float fDen = fDenC + fDenX * fX + fDenY * fY;
				const float fr = __fdividef(1.0f, fDen);

				const float fT = fNum * fr;
				fVal += tex2D(gT_FanProjTexture, fT, fA) * fr;
				fY -= fSubStep;
			}
			fX += fSubStep;
		}
		fA += 1.0f;
	}

	volData[Y*volPitch+X] += fVal * fOutputScale;
}


// BP specifically for SART.
// It includes (free) weighting with voxel weight.
// It assumes the proj texture is set up _without_ padding, unlike regular BP.
__global__ void devFanBP_SART(float* D_volData, unsigned int volPitch, const SDimensions dims, float fOutputScale)
{
	const int relX = threadIdx.x;
	const int relY = threadIdx.y;

	const int X = blockIdx.x * g_blockSlices + relX;
	const int Y = blockIdx.y * g_blockSliceSize + relY;

	if (X >= dims.iVolWidth || Y >= dims.iVolHeight)
		return;

	const float fX = ( X - 0.5f*dims.iVolWidth + 0.5f );
	const float fY = - ( Y - 0.5f*dims.iVolHeight + 0.5f );

	float* volData = (float*)D_volData;

	const float fNumC = gC_C[0].fNumC;
	const float fNumX = gC_C[0].fNumX;
	const float fNumY = gC_C[0].fNumY;
	const float fDenC = gC_C[0].fDenC;
	const float fDenX = gC_C[0].fDenX;
	const float fDenY = gC_C[0].fDenY;

	const float fNum = fNumC + fNumX * fX + fNumY * fY;
	const float fDen = fDenC + fDenX * fX + fDenY * fY;

	const float fr = __fdividef(1.0f, fDen);
	const float fT = fNum * fr;
	// NB: The scale constant in devBP is cancelled out by the SART weighting
	const float fVal = tex2D(gT_FanProjTexture, fT, 0.5f);

	volData[Y*volPitch+X] += fVal * fOutputScale;
}

struct Vec2 {
	double x;
	double y;
	Vec2(double x_, double y_) : x(x_), y(y_) { }
	Vec2 operator+(const Vec2 &b) const {
		return Vec2(x + b.x, y + b.y);
	}
	Vec2 operator-(const Vec2 &b) const {
		return Vec2(x - b.x, y - b.y);
	}
	Vec2 operator-() const {
		return Vec2(-x, -y);
	}
	double norm() const {
		return sqrt(x*x + y*y);
	}
};

double det2(const Vec2 &a, const Vec2 &b) {
	return a.x * b.y - a.y * b.x;
}


bool transferConstants(const SFanProjection* angles, unsigned int iProjAngles, bool FBP)
{
	DevFanParams *p = new DevFanParams[iProjAngles];

	// We need three values in the kernel:
	// projected coordinates of pixels on the detector:
	// || x (s-d) || + ||s d|| / || u (s-x) ||

	// ray density weighting factor for the adjoint
	// || u (s-d) || / ( |u| * || u (s-x) || )

	// fan-beam FBP weighting factor
	// ( || u s || / || u (s-x) || ) ^ 2



	for (unsigned int i = 0; i < iProjAngles; ++i) {
		Vec2 u(angles[i].fDetUX, angles[i].fDetUY);
		Vec2 s(angles[i].fSrcX, angles[i].fSrcY);
		Vec2 d(angles[i].fDetSX, angles[i].fDetSY);



		double fScale;
		if (!FBP) {
			// goal: 1/fDen = || u (s-d) || / ( |u| * || u (s-x) || )
			// fDen = ( |u| * || u (s-x) || ) / || u (s-d) ||
			// i.e. scale = |u| /  || u (s-d) ||

			fScale = u.norm() / det2(u, s-d);
		} else {
			// goal: 1/fDen = || u s || / || u (s-x) ||
			// fDen = || u (s-x) || / || u s ||
			// i.e., scale = 1 / || u s ||

			fScale = 1.0 / det2(u, s);
		}

		p[i].fNumC = fScale * det2(s,d);
		p[i].fNumX = fScale * (s-d).y;
		p[i].fNumY = -fScale * (s-d).x;
		p[i].fDenC = fScale * det2(u, s); // == 1.0 for FBP
		p[i].fDenX = fScale * u.y;
		p[i].fDenY = -fScale * u.x;
	}

	// TODO: Check for errors
	cudaMemcpyToSymbol(gC_C, p, iProjAngles*sizeof(DevFanParams), 0, cudaMemcpyHostToDevice);

	return true;
}


bool FanBP_internal(float* D_volumeData, unsigned int volumePitch,
           float* D_projData, unsigned int projPitch,
           const SDimensions& dims, const SFanProjection* angles,
           float fOutputScale)
{
	assert(dims.iProjAngles <= g_MaxAngles);

	bindProjDataTexture(D_projData, projPitch, dims.iProjDets, dims.iProjAngles);

	bool ok = transferConstants(angles, dims.iProjAngles, false);
	if (!ok)
		return false;

	dim3 dimBlock(g_blockSlices, g_blockSliceSize);
	dim3 dimGrid((dims.iVolWidth+g_blockSlices-1)/g_blockSlices,
	             (dims.iVolHeight+g_blockSliceSize-1)/g_blockSliceSize);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	for (unsigned int i = 0; i < dims.iProjAngles; i += g_anglesPerBlock) {
		if (dims.iRaysPerPixelDim > 1)
			devFanBP_SS<<<dimGrid, dimBlock, 0, stream>>>(D_volumeData, volumePitch, i, dims, fOutputScale);
		else
			devFanBP<false><<<dimGrid, dimBlock, 0, stream>>>(D_volumeData, volumePitch, i, dims, fOutputScale);
	}
	cudaThreadSynchronize();

	cudaTextForceKernelsCompletion();

	cudaStreamDestroy(stream);

	return true;
}

bool FanBP_FBPWeighted_internal(float* D_volumeData, unsigned int volumePitch,
           float* D_projData, unsigned int projPitch,
           const SDimensions& dims, const SFanProjection* angles,
           float fOutputScale)
{
	assert(dims.iProjAngles <= g_MaxAngles);

	bindProjDataTexture(D_projData, projPitch, dims.iProjDets, dims.iProjAngles);

	bool ok = transferConstants(angles, dims.iProjAngles, true);
	if (!ok)
		return false;

	dim3 dimBlock(g_blockSlices, g_blockSliceSize);
	dim3 dimGrid((dims.iVolWidth+g_blockSlices-1)/g_blockSlices,
	             (dims.iVolHeight+g_blockSliceSize-1)/g_blockSliceSize);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	for (unsigned int i = 0; i < dims.iProjAngles; i += g_anglesPerBlock) {
		devFanBP<true><<<dimGrid, dimBlock, 0, stream>>>(D_volumeData, volumePitch, i, dims, fOutputScale);
	}
	cudaThreadSynchronize();

	cudaTextForceKernelsCompletion();

	cudaStreamDestroy(stream);

	return true;
}

// D_projData is a pointer to one padded sinogram line
bool FanBP_SART(float* D_volumeData, unsigned int volumePitch,
                float* D_projData, unsigned int projPitch,
                unsigned int angle,
                const SDimensions& dims, const SFanProjection* angles,
                float fOutputScale)
{
	// only one angle
	bindProjDataTexture(D_projData, projPitch, dims.iProjDets, 1, cudaAddressModeClamp);

	bool ok = transferConstants(angles + angle, 1, false);
	if (!ok)
		return false;

	dim3 dimBlock(g_blockSlices, g_blockSliceSize);
	dim3 dimGrid((dims.iVolWidth+g_blockSlices-1)/g_blockSlices,
	             (dims.iVolHeight+g_blockSliceSize-1)/g_blockSliceSize);

	devFanBP_SART<<<dimGrid, dimBlock>>>(D_volumeData, volumePitch, dims, fOutputScale);
	cudaThreadSynchronize();

	cudaTextForceKernelsCompletion();

	return true;
}

bool FanBP(float* D_volumeData, unsigned int volumePitch,
           float* D_projData, unsigned int projPitch,
           const SDimensions& dims, const SFanProjection* angles,
           float fOutputScale)
{
	for (unsigned int iAngle = 0; iAngle < dims.iProjAngles; iAngle += g_MaxAngles) {
		SDimensions subdims = dims;
		unsigned int iEndAngle = iAngle + g_MaxAngles;
		if (iEndAngle >= dims.iProjAngles)
			iEndAngle = dims.iProjAngles;
		subdims.iProjAngles = iEndAngle - iAngle;

		bool ret;
		ret = FanBP_internal(D_volumeData, volumePitch,
		                  D_projData + iAngle * projPitch, projPitch,
		                  subdims, angles + iAngle, fOutputScale);
		if (!ret)
			return false;
	}
	return true;
}

bool FanBP_FBPWeighted(float* D_volumeData, unsigned int volumePitch,
           float* D_projData, unsigned int projPitch,
           const SDimensions& dims, const SFanProjection* angles,
           float fOutputScale)
{
	for (unsigned int iAngle = 0; iAngle < dims.iProjAngles; iAngle += g_MaxAngles) {
		SDimensions subdims = dims;
		unsigned int iEndAngle = iAngle + g_MaxAngles;
		if (iEndAngle >= dims.iProjAngles)
			iEndAngle = dims.iProjAngles;
		subdims.iProjAngles = iEndAngle - iAngle;

		bool ret;
		ret = FanBP_FBPWeighted_internal(D_volumeData, volumePitch,
		                  D_projData + iAngle * projPitch, projPitch,
		                  subdims, angles + iAngle, fOutputScale);

		if (!ret)
			return false;
	}
	return true;
}


}
