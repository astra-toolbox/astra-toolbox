/*
-----------------------------------------------------------------------
Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
           2014-2015, CWI, Amsterdam

Contact: astra@uantwerpen.be
Website: http://sf.net/projects/astra-toolbox

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
$Id$
*/

#include <cstdio>
#include <cassert>
#include <iostream>
#include <list>

#include <cuda.h>
#include "util3d.h"

#ifdef STANDALONE
#include "cone_fp.h"
#include "testutil.h"
#endif

#include "dims3d.h"
#include "arith3d.h"
#include "../2d/fft.h"

typedef texture<float, 3, cudaReadModeElementType> texture3D;

static texture3D gT_coneProjTexture;

namespace astraCUDA3d {

static const unsigned int g_volBlockZ = 16;

static const unsigned int g_anglesPerBlock = 64;
static const unsigned int g_volBlockX = 32;
static const unsigned int g_volBlockY = 16;

static const unsigned int g_anglesPerWeightBlock = 16;
static const unsigned int g_detBlockU = 32;
static const unsigned int g_detBlockV = 32;

static const unsigned g_MaxAngles = 2048;

__constant__ float gC_angle_sin[g_MaxAngles];
__constant__ float gC_angle_cos[g_MaxAngles];
__constant__ float gC_angle[g_MaxAngles];


// per-detector u/v shifts?

static bool bindProjDataTexture(const cudaArray* array)
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


__global__ void devBP_FDK(void* D_volData, unsigned int volPitch, int startAngle, float fSrcOrigin, float fDetOrigin, float fSrcZ, float fDetZ, float fInvDetUSize, float fInvDetVSize, const SDimensions3D dims)
{
	float* volData = (float*)D_volData;

	int endAngle = startAngle + g_anglesPerBlock;
	if (endAngle > dims.iProjAngles)
		endAngle = dims.iProjAngles;

	// threadIdx: x = rel x
	//            y = rel y

	// blockIdx:  x = x + y
    //            y = z


	// TO TRY: precompute part of detector intersection formulas in shared mem?
	// TO TRY: inner loop over z, gather ray values in shared mem

	const int X = blockIdx.x % ((dims.iVolX+g_volBlockX-1)/g_volBlockX) * g_volBlockX + threadIdx.x;
	const int Y = blockIdx.x / ((dims.iVolX+g_volBlockX-1)/g_volBlockX) * g_volBlockY + threadIdx.y;

	if (X > dims.iVolX)
		return;
	if (Y > dims.iVolY)
		return;

	const int startZ = blockIdx.y * g_volBlockZ;
	int endZ = startZ + g_volBlockZ;
	if (endZ > dims.iVolZ)
		endZ = dims.iVolZ;

	float fX = X - 0.5f*dims.iVolX + 0.5f;
	float fY = Y - 0.5f*dims.iVolY + 0.5f;
	float fZ = startZ - 0.5f*dims.iVolZ + 0.5f - fSrcZ;

	const float fU_base = 0.5f*dims.iProjU - 0.5f + 0.5f;
	const float fV_base = 0.5f*dims.iProjV - 0.5f + 0.5f + (fDetZ-fSrcZ);

	// Note re. fZ/rV_base: the computations below are all relative to the
	// optical axis, so we do the Z-adjustments beforehand.

	for (int Z = startZ; Z < endZ; ++Z, fZ += 1.0f)
	{

		float fVal = 0.0f;
		float fAngle = startAngle + 0.5f;

		for (int angle = startAngle; angle < endAngle; ++angle, fAngle += 1.0f)
		{

			const float cos_theta = gC_angle_cos[angle];
			const float sin_theta = gC_angle_sin[angle];

			const float fR = fSrcOrigin;
			const float fD = fR - fX * sin_theta + fY * cos_theta;
			float fWeight = fR / fD;
			fWeight *= fWeight;

			const float fScaleFactor = (fR + fDetOrigin) / fD;
			const float fU = fU_base + (fX*cos_theta+fY*sin_theta) * fScaleFactor * fInvDetUSize;
			const float fV = fV_base + fZ * fScaleFactor * fInvDetVSize;

			fVal += tex3D(gT_coneProjTexture, fU, fAngle, fV);

		}

		volData[(Z*dims.iVolY+Y)*volPitch+X] += fVal;
//		projData[(angle*dims.iProjV+detectorV)*projPitch+detectorU] = 10.0f;
//		if (threadIdx.x == 0 && threadIdx.y == 0) { printf("%d,%d,%d [%d / %d] -> %f\n", angle, detectorU, detectorV, (angle*dims.iProjV+detectorV)*projPitch+detectorU, projPitch, projData[(angle*dims.iProjV+detectorV)*projPitch+detectorU]); }
	}

}


bool FDK_BP(cudaPitchedPtr D_volumeData,
            cudaPitchedPtr D_projData,
            float fSrcOrigin, float fDetOrigin,
            float fSrcZ, float fDetZ, float fDetUSize, float fDetVSize,
            const SDimensions3D& dims, const float* angles)
{
	// transfer projections to array

	cudaArray* cuArray = allocateProjectionArray(dims);
	transferProjectionsToArray(D_projData, cuArray, dims);

	bindProjDataTexture(cuArray);

	float* angle_sin = new float[dims.iProjAngles];
	float* angle_cos = new float[dims.iProjAngles];

	for (unsigned int i = 0; i < dims.iProjAngles; ++i) {
		angle_sin[i] = sinf(angles[i]);
		angle_cos[i] = cosf(angles[i]);
	}
	cudaError_t e1 = cudaMemcpyToSymbol(gC_angle_sin, angle_sin, dims.iProjAngles*sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaError_t e2 = cudaMemcpyToSymbol(gC_angle_cos, angle_cos, dims.iProjAngles*sizeof(float), 0, cudaMemcpyHostToDevice);
	assert(e1 == cudaSuccess);
	assert(e2 == cudaSuccess);

	delete[] angle_sin;
	delete[] angle_cos;

	dim3 dimBlock(g_volBlockX, g_volBlockY);

	dim3 dimGrid(((dims.iVolX+g_volBlockX-1)/g_volBlockX)*((dims.iVolY+g_volBlockY-1)/g_volBlockY), (dims.iVolZ+g_volBlockZ-1)/g_volBlockZ);

	// timeval t;
	// tic(t);

	for (unsigned int i = 0; i < dims.iProjAngles; i += g_anglesPerBlock) {
		devBP_FDK<<<dimGrid, dimBlock>>>(D_volumeData.ptr, D_volumeData.pitch/sizeof(float), i, fSrcOrigin, fDetOrigin, fSrcZ, fDetZ, 1.0f / fDetUSize, 1.0f / fDetVSize, dims);
	}

	cudaTextForceKernelsCompletion();

	cudaFreeArray(cuArray);

	// printf("%f\n", toc(t));

	return true;
}

__global__ void devFDK_preweight(void* D_projData, unsigned int projPitch, unsigned int startAngle, unsigned int endAngle, float fSrcOrigin, float fDetOrigin, float fSrcZ, float fDetZ, float fDetUSize, float fDetVSize, const SDimensions3D dims)
{
	float* projData = (float*)D_projData;
	int angle = startAngle + blockIdx.y * g_anglesPerWeightBlock + threadIdx.y;
	if (angle >= endAngle)
		return;

	const int detectorU = (blockIdx.x%((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockU + threadIdx.x;
	const int startDetectorV = (blockIdx.x/((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockV;
	int endDetectorV = startDetectorV + g_detBlockV;
	if (endDetectorV > dims.iProjV)
		endDetectorV = dims.iProjV;

	// We need the length of the central ray and the length of the ray(s) to
	// our detector pixel(s).

	const float fCentralRayLength = fSrcOrigin + fDetOrigin;

	const float fU = (detectorU - 0.5f*dims.iProjU + 0.5f) * fDetUSize;

	const float fT = fCentralRayLength * fCentralRayLength + fU * fU;

	float fV = (startDetectorV - 0.5f*dims.iProjV + 0.5f) * fDetVSize + fDetZ - fSrcZ;

	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)
	{
		const float fRayLength = sqrtf(fT + fV * fV);

		const float fWeight = fCentralRayLength / fRayLength;

		projData[(detectorV*dims.iProjAngles+angle)*projPitch+detectorU] *= fWeight;

		fV += 1.0f;
	}
}

__global__ void devFDK_ParkerWeight(void* D_projData, unsigned int projPitch, unsigned int startAngle, unsigned int endAngle, float fSrcOrigin, float fDetOrigin, float fSrcZ, float fDetZ, float fDetUSize, float fCentralFanAngle, const SDimensions3D dims)
{
	float* projData = (float*)D_projData;
	int angle = startAngle + blockIdx.y * g_anglesPerWeightBlock + threadIdx.y;
	if (angle >= endAngle)
		return;

	const int detectorU = (blockIdx.x%((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockU + threadIdx.x;
	const int startDetectorV = (blockIdx.x/((dims.iProjU+g_detBlockU-1)/g_detBlockU)) * g_detBlockV;
	int endDetectorV = startDetectorV + g_detBlockV;
	if (endDetectorV > dims.iProjV)
		endDetectorV = dims.iProjV;

	// We need the length of the central ray and the length of the projection
	// of our ray onto the central slice

	const float fCentralRayLength = fSrcOrigin + fDetOrigin;

	// TODO: Detector pixel size
	const float fU = (detectorU - 0.5f*dims.iProjU + 0.5f) * fDetUSize;

	//const float fGamma = atanf(fU / fCentralRayLength);
	//const float fBeta = gC_angle[angle];
	const float fGamma = atanf(fU / fCentralRayLength);
	float fBeta = -gC_angle[angle];
	if (fBeta < 0.0f)
		fBeta += 2*M_PI;
	if (fBeta >= 2*M_PI)
		fBeta -= 2*M_PI;

	// compute the weight depending on the location in the central fan's radon
	// space
	float fWeight;

	if (fBeta <= 0.0f) {
		fWeight = 0.0f;
	} else if (fBeta <= 2.0f*(fCentralFanAngle + fGamma)) {
		fWeight = sinf((M_PI / 4.0f) * fBeta / (fCentralFanAngle + fGamma));
		fWeight *= fWeight;
	} else if (fBeta <= M_PI + 2*fGamma) {
		fWeight = 1.0f;
	} else if (fBeta <= M_PI + 2*fCentralFanAngle) {
		fWeight = sinf((M_PI / 4.0f) * (M_PI + 2.0f*fCentralFanAngle - fBeta) / (fCentralFanAngle - fGamma));
		fWeight *= fWeight;
	} else {
		fWeight = 0.0f;
	}

	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)
	{

		projData[(detectorV*dims.iProjAngles+angle)*projPitch+detectorU] *= fWeight;

	}
}



// Perform the FDK pre-weighting and filtering
bool FDK_PreWeight(cudaPitchedPtr D_projData,
                float fSrcOrigin, float fDetOrigin,
                float fSrcZ, float fDetZ,
                float fDetUSize, float fDetVSize, bool bShortScan,
                const SDimensions3D& dims, const float* angles)
{
	// The pre-weighting factor for a ray is the cosine of the angle between
	// the central line and the ray.

	dim3 dimBlock(g_detBlockU, g_anglesPerWeightBlock);
	dim3 dimGrid( ((dims.iProjU+g_detBlockU-1)/g_detBlockU)*((dims.iProjV+g_detBlockV-1)/g_detBlockV),
	              (dims.iProjAngles+g_anglesPerWeightBlock-1)/g_anglesPerWeightBlock);

	int projPitch = D_projData.pitch/sizeof(float);

	devFDK_preweight<<<dimGrid, dimBlock>>>(D_projData.ptr, projPitch, 0, dims.iProjAngles, fSrcOrigin, fDetOrigin, fSrcZ, fDetZ, fDetUSize, fDetVSize, dims);

	cudaTextForceKernelsCompletion();

	if (bShortScan) {
		// We do short-scan Parker weighting

		cudaError_t e1 = cudaMemcpyToSymbol(gC_angle, angles,
		                                    dims.iProjAngles*sizeof(float), 0,
		                                    cudaMemcpyHostToDevice);
		assert(!e1);

		// TODO: detector pixel size!
		float fCentralFanAngle = atanf((dims.iProjU*0.5f) /
		                               (fSrcOrigin + fDetOrigin));

		devFDK_ParkerWeight<<<dimGrid, dimBlock>>>(D_projData.ptr, projPitch, 0, dims.iProjAngles, fSrcOrigin, fDetOrigin, fSrcZ, fDetZ, fDetUSize, fCentralFanAngle, dims);

	}

	cudaTextForceKernelsCompletion();
	return true;
}

bool FDK_Filter(cudaPitchedPtr D_projData,
                cufftComplex * D_filter,
                float fSrcOrigin, float fDetOrigin,
                float fSrcZ, float fDetZ,
                float fDetUSize, float fDetVSize, bool bShortScan,
                const SDimensions3D& dims, const float* angles)
{

	// The filtering is a regular ramp filter per detector line.

	int iPaddedDetCount = calcNextPowerOfTwo(2 * dims.iProjU);
	int iHalfFFTSize = calcFFTFourSize(iPaddedDetCount);
	int projPitch = D_projData.pitch/sizeof(float);
	

	// We process one sinogram at a time.
	float* D_sinoData = (float*)D_projData.ptr;

	cufftComplex * D_sinoFFT = NULL;
	allocateComplexOnDevice(dims.iProjAngles, iHalfFFTSize, &D_sinoFFT);

	bool ok = true;

	for (int v = 0; v < dims.iProjV; ++v) {

		ok = runCudaFFT(dims.iProjAngles, D_sinoData, projPitch,
		                dims.iProjU, iPaddedDetCount, iHalfFFTSize,
		                D_sinoFFT);

		if (!ok) break;

		applyFilter(dims.iProjAngles, iHalfFFTSize, D_sinoFFT, D_filter);


		ok = runCudaIFFT(dims.iProjAngles, D_sinoFFT, D_sinoData, projPitch,
		                 dims.iProjU, iPaddedDetCount, iHalfFFTSize);

		if (!ok) break;

		D_sinoData += (dims.iProjAngles * projPitch);
	}

	freeComplexOnDevice(D_sinoFFT);

	return ok;
}


bool FDK(cudaPitchedPtr D_volumeData,
         cudaPitchedPtr D_projData,
         float fSrcOrigin, float fDetOrigin,
         float fSrcZ, float fDetZ, float fDetUSize, float fDetVSize,
         const SDimensions3D& dims, const float* angles, bool bShortScan)
{
	bool ok;
	// Generate filter
	// TODO: Check errors
	cufftComplex * D_filter;
	int iPaddedDetCount = calcNextPowerOfTwo(2 * dims.iProjU);
	int iHalfFFTSize = calcFFTFourSize(iPaddedDetCount);

	ok = FDK_PreWeight(D_projData, fSrcOrigin, fDetOrigin,
	                fSrcZ, fDetZ, fDetUSize, fDetVSize,
	                bShortScan, dims, angles);
	if (!ok)
		return false;

	cufftComplex *pHostFilter = new cufftComplex[dims.iProjAngles * iHalfFFTSize];
	memset(pHostFilter, 0, sizeof(cufftComplex) * dims.iProjAngles * iHalfFFTSize);

	genFilter(FILTER_RAMLAK, 1.0f, dims.iProjAngles, pHostFilter, iPaddedDetCount, iHalfFFTSize);


	allocateComplexOnDevice(dims.iProjAngles, iHalfFFTSize, &D_filter);
	uploadComplexArrayToDevice(dims.iProjAngles, iHalfFFTSize, pHostFilter, D_filter);

	delete [] pHostFilter;


	// Perform filtering



	ok = FDK_Filter(D_projData, D_filter, fSrcOrigin, fDetOrigin,
	                fSrcZ, fDetZ, fDetUSize, fDetVSize,
	                bShortScan, dims, angles);

	// Clean up filter
	freeComplexOnDevice(D_filter);


	if (!ok)
		return false;

	// Perform BP

	ok = FDK_BP(D_volumeData, D_projData, fSrcOrigin, fDetOrigin, fSrcZ, fDetZ,
	            fDetUSize, fDetVSize, dims, angles);

	if (!ok)
		return false;

	processVol3D<opMul>(D_volumeData,
	                  (M_PI / 2.0f) / (float)dims.iProjAngles, dims);

	return true;
}


}

#ifdef STANDALONE
void dumpVolume(const char* filespec, const cudaPitchedPtr& data, const SDimensions3D& dims, float fMin, float fMax)
{
	float* buf = new float[dims.iVolX*dims.iVolY];
	unsigned int pitch = data.pitch / sizeof(float);

	for (int i = 0; i < dims.iVolZ; ++i) {
		cudaMemcpy2D(buf, dims.iVolX*sizeof(float), ((float*)data.ptr)+pitch*dims.iVolY*i, data.pitch, dims.iVolX*sizeof(float), dims.iVolY, cudaMemcpyDeviceToHost);

		char fname[512];
		sprintf(fname, filespec, dims.iVolZ-i-1);
		saveImage(fname, dims.iVolY, dims.iVolX, buf, fMin, fMax);
	}
}

void dumpSinograms(const char* filespec, const cudaPitchedPtr& data, const SDimensions3D& dims, float fMin, float fMax)
{
	float* bufs = new float[dims.iProjAngles*dims.iProjU];
	unsigned int pitch = data.pitch / sizeof(float);

	for (int i = 0; i < dims.iProjV; ++i) {
		cudaMemcpy2D(bufs, dims.iProjU*sizeof(float), ((float*)data.ptr)+pitch*dims.iProjAngles*i, data.pitch, dims.iProjU*sizeof(float), dims.iProjAngles, cudaMemcpyDeviceToHost);

		char fname[512];
		sprintf(fname, filespec, i);
		saveImage(fname, dims.iProjAngles, dims.iProjU, bufs, fMin, fMax);
	}
}

void dumpProjections(const char* filespec, const cudaPitchedPtr& data, const SDimensions3D& dims, float fMin, float fMax)
{
	float* bufp = new float[dims.iProjV*dims.iProjU];
	unsigned int pitch = data.pitch / sizeof(float);

	for (int i = 0; i < dims.iProjAngles; ++i) {
		for (int j = 0; j < dims.iProjV; ++j) {
			cudaMemcpy(bufp+dims.iProjU*j, ((float*)data.ptr)+pitch*dims.iProjAngles*j+pitch*i, dims.iProjU*sizeof(float), cudaMemcpyDeviceToHost);
		}

		char fname[512];
		sprintf(fname, filespec, i);
		saveImage(fname, dims.iProjV, dims.iProjU, bufp, fMin, fMax);
	}
}




int main()
{
#if 0
	SDimensions3D dims;
	dims.iVolX = 512;
	dims.iVolY = 512;
	dims.iVolZ = 512;
	dims.iProjAngles = 180;
	dims.iProjU = 1024;
	dims.iProjV = 1024;
	dims.iRaysPerDet = 1;

	cudaExtent extentV;
	extentV.width = dims.iVolX*sizeof(float);
	extentV.height = dims.iVolY;
	extentV.depth = dims.iVolZ;

	cudaPitchedPtr volData; // pitch, ptr, xsize, ysize

	cudaMalloc3D(&volData, extentV);

	cudaExtent extentP;
	extentP.width = dims.iProjU*sizeof(float);
	extentP.height = dims.iProjAngles;
	extentP.depth = dims.iProjV;

	cudaPitchedPtr projData; // pitch, ptr, xsize, ysize

	cudaMalloc3D(&projData, extentP);
	cudaMemset3D(projData, 0, extentP);

#if 0
	float* slice = new float[256*256];
	cudaPitchedPtr ptr;
	ptr.ptr = slice;
	ptr.pitch = 256*sizeof(float);
	ptr.xsize = 256*sizeof(float);
	ptr.ysize = 256;

	for (unsigned int i = 0; i < 256*256; ++i)
		slice[i] = 1.0f;
	for (unsigned int i = 0; i < 256; ++i) {
		cudaExtent extentS;
		extentS.width = dims.iVolX*sizeof(float);
		extentS.height = dims.iVolY;
		extentS.depth = 1;
		cudaPos sp = { 0, 0, 0 };
		cudaPos dp = { 0, 0, i };
		cudaMemcpy3DParms p;
		p.srcArray = 0;
		p.srcPos = sp;
		p.srcPtr = ptr;
		p.dstArray = 0;
		p.dstPos = dp;
		p.dstPtr = volData;
		p.extent = extentS;
		p.kind = cudaMemcpyHostToDevice;
		cudaMemcpy3D(&p);
#if 0
		if (i == 128) {
			for (unsigned int j = 0; j < 256*256; ++j)
				slice[j] = 0.0f;
		}
#endif 
	}
#endif

	SConeProjection angle[180];
	angle[0].fSrcX = -1536;
	angle[0].fSrcY = 0;
	angle[0].fSrcZ = 0;

	angle[0].fDetSX = 1024;
	angle[0].fDetSY = -512;
	angle[0].fDetSZ = 512;

	angle[0].fDetUX = 0;
	angle[0].fDetUY = 1;
	angle[0].fDetUZ = 0;

	angle[0].fDetVX = 0;
	angle[0].fDetVY = 0;
	angle[0].fDetVZ = -1;

#define ROTATE0(name,i,alpha) do { angle[i].f##name##X = angle[0].f##name##X * cos(alpha) - angle[0].f##name##Y * sin(alpha); angle[i].f##name##Y = angle[0].f##name##X * sin(alpha) + angle[0].f##name##Y * cos(alpha); } while(0)
	for (int i = 1; i < 180; ++i) {
		angle[i] = angle[0];
		ROTATE0(Src, i, i*2*M_PI/180);
		ROTATE0(DetS, i, i*2*M_PI/180);
		ROTATE0(DetU, i, i*2*M_PI/180);
		ROTATE0(DetV, i, i*2*M_PI/180);
	}
#undef ROTATE0

	astraCUDA3d::ConeFP(volData, projData, dims, angle, 1.0f);

	//dumpSinograms("sino%03d.png", projData, dims, 0, 512);
	//dumpProjections("proj%03d.png", projData, dims, 0, 512);

	astraCUDA3d::zeroVolumeData(volData, dims);

	float* angles = new float[dims.iProjAngles];
	for (int i = 0; i < 180; ++i)
		angles[i] = i*2*M_PI/180;

	astraCUDA3d::FDK(volData, projData, 1536, 512, 0, 0, dims, angles);

	dumpVolume("vol%03d.png", volData, dims, -20, 100);


#else

	SDimensions3D dims;
	dims.iVolX = 1000;
	dims.iVolY = 999;
	dims.iVolZ = 500;
	dims.iProjAngles = 376;
	dims.iProjU = 1024;
	dims.iProjV = 524;
	dims.iRaysPerDet = 1;

	float* angles = new float[dims.iProjAngles];
	for (int i = 0; i < dims.iProjAngles; ++i)
		angles[i] = -i*(M_PI)/360;

	cudaPitchedPtr volData = astraCUDA3d::allocateVolumeData(dims);
	cudaPitchedPtr projData = astraCUDA3d::allocateProjectionData(dims);
	astraCUDA3d::zeroProjectionData(projData, dims);
	astraCUDA3d::zeroVolumeData(volData, dims);

	timeval t;
	tic(t);

	for (int i = 0; i < dims.iProjAngles; ++i) {
		char fname[256];
		sprintf(fname, "/home/wpalenst/tmp/Elke/proj%04d.png", i);
		unsigned int w,h;
		float* bufp = loadImage(fname, w,h);

		int pitch = projData.pitch / sizeof(float);
		for (int j = 0; j < dims.iProjV; ++j) {
			cudaMemcpy(((float*)projData.ptr)+dims.iProjAngles*pitch*j+pitch*i, bufp+dims.iProjU*j, dims.iProjU*sizeof(float), cudaMemcpyHostToDevice);
		}

		delete[] bufp;
	}
	printf("Load time: %f\n", toc(t));

	//dumpSinograms("sino%03d.png", projData, dims, -8.0f, 256.0f);
	//astraCUDA3d::FDK(volData, projData, 7350, 62355, 0, 10, dims, angles);
	//astraCUDA3d::FDK(volData, projData, 7350, -380, 0, 10, dims, angles);

	tic(t);

	astraCUDA3d::FDK(volData, projData, 7383.29867, 0, 0, 10, dims, angles);

	printf("FDK time: %f\n", toc(t));
	tic(t);

	dumpVolume("vol%03d.png", volData, dims, -65.9f, 200.0f);
	//dumpVolume("vol%03d.png", volData, dims, 0.0f, 256.0f);
	printf("Save time: %f\n", toc(t));

#endif


}
#endif
