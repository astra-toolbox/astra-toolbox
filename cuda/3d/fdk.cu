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
#include "astra/cuda/3d/arith3d.h"
#include "astra/cuda/3d/cone_bp.h"

#include "astra/cuda/2d/fft.h"

#ifdef STANDALONE
#include "astra/cuda/3d/cone_fp.h"
#include "testutil.h"
#endif

#include "astra/Logging.h"

#include <cstdio>
#include <cassert>
#include <iostream>
#include <list>

#include <cuda.h>

namespace astraCUDA3d {

static const unsigned int g_anglesPerWeightBlock = 16;
static const unsigned int g_detBlockU = 32;
static const unsigned int g_detBlockV = 32;

static const unsigned g_MaxAngles = 12000;

__constant__ float gC_angle[g_MaxAngles];


// per-detector u/v shifts?


__global__ void devFDK_preweight(void* D_projData, unsigned int projPitch, unsigned int startAngle, unsigned int endAngle, float fSrcOrigin, float fDetOrigin, float fZShift, float fDetUSize, float fDetVSize, float fVoxSize, const SDimensions3D dims)
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

	float fV = (startDetectorV - 0.5f*dims.iProjV + 0.5f) * fDetVSize + fZShift;

	// Contributions to the weighting factors:
	// fCentralRayLength / fRayLength   : the main FDK preweighting factor
	// fSrcOrigin / (fDetUSize * fCentralRayLength)
	//                                  : to adjust the filter to the det width
	// || u v s || ^ 2                  : see cone_bp.cu, FDKWEIGHT
	// pi / (2 * iProjAngles)           : scaling of the integral over angles
	// fVoxSize ^ 2                     : ...

	const float fW1 = fSrcOrigin * fDetUSize * fDetVSize;
	const float fW2 = fCentralRayLength / (fDetUSize * fSrcOrigin);
	const float fW3 = fVoxSize * fVoxSize;
	const float fW = fCentralRayLength * fW1 * fW1 * fW2 * fW3 * (M_PI / 2.0f) / (float)dims.iProjAngles;

	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)
	{
		const float fRayLength = sqrtf(fT + fV * fV);

		const float fWeight = fW / fRayLength;

		projData[(detectorV*dims.iProjAngles+angle)*projPitch+detectorU] *= fWeight;

		fV += fDetVSize;
	}
}

__global__ void devFDK_ParkerWeight(void* D_projData, unsigned int projPitch, unsigned int startAngle, unsigned int endAngle, float fSrcOrigin, float fDetOrigin, float fDetUSize, float fCentralFanAngle, const SDimensions3D dims)
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

	const float fGamma = atanf(fU / fCentralRayLength);
	float fBeta = gC_angle[angle];

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

	fWeight *= 2; // adjust to effectively halved angular range

	for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV)
	{

		projData[(detectorV*dims.iProjAngles+angle)*projPitch+detectorU] *= fWeight;

	}
}



// Perform the FDK pre-weighting and filtering
bool FDK_PreWeight(cudaPitchedPtr D_projData,
                float fSrcOrigin, float fDetOrigin,
                float fZShift,
                float fDetUSize, float fDetVSize, float fVoxSize,
				bool bShortScan,
                const SDimensions3D& dims, const float* angles)
{
	// The pre-weighting factor for a ray is the cosine of the angle between
	// the central line and the ray.

	dim3 dimBlock(g_detBlockU, g_anglesPerWeightBlock);
	dim3 dimGrid( ((dims.iProjU+g_detBlockU-1)/g_detBlockU)*((dims.iProjV+g_detBlockV-1)/g_detBlockV),
	              (dims.iProjAngles+g_anglesPerWeightBlock-1)/g_anglesPerWeightBlock);

	int projPitch = D_projData.pitch/sizeof(float);

	devFDK_preweight<<<dimGrid, dimBlock>>>(D_projData.ptr, projPitch, 0, dims.iProjAngles, fSrcOrigin, fDetOrigin, fZShift, fDetUSize, fDetVSize, fVoxSize, dims);

	cudaTextForceKernelsCompletion();

	if (bShortScan && dims.iProjAngles > 1) {
		ASTRA_DEBUG("Doing Parker weighting");
		// We do short-scan Parker weighting

		// First, determine (in a very basic way) the interval that's
		// been scanned. We assume angles[0] is one of the endpoints of the
		// range.
		float fdA = angles[1] - angles[0];

		while (fdA < -M_PI)
			fdA += 2*M_PI;
		while (fdA >= M_PI)
			fdA -= 2*M_PI;

		float fAngleBase;
		if (fdA >= 0.0f) {
			// going up from angles[0]
			fAngleBase = angles[0];
		} else {
			// going down from angles[0]
			fAngleBase = angles[dims.iProjAngles - 1];
		}

		// We pick the lowest end of the range, and then
		// move all angles so they fall in [0,2pi)

		float *fRelAngles = new float[dims.iProjAngles];
		for (unsigned int i = 0; i < dims.iProjAngles; ++i) {
			float f = angles[i] - fAngleBase;
			while (f >= 2*M_PI)
				f -= 2*M_PI;
			while (f < 0)
				f += 2*M_PI;
			fRelAngles[i] = f;
		}

		cudaError_t e1 = cudaMemcpyToSymbol(gC_angle, fRelAngles,
		                                    dims.iProjAngles*sizeof(float), 0,
		                                    cudaMemcpyHostToDevice);
		assert(!e1);
		delete[] fRelAngles;

		float fCentralFanAngle = atanf(fDetUSize * (dims.iProjU*0.5f) /
		                               (fSrcOrigin + fDetOrigin));

		devFDK_ParkerWeight<<<dimGrid, dimBlock>>>(D_projData.ptr, projPitch, 0, dims.iProjAngles, fSrcOrigin, fDetOrigin, fDetUSize, fCentralFanAngle, dims);

	}

	cudaTextForceKernelsCompletion();
	return true;
}

bool FDK_Filter(cudaPitchedPtr D_projData,
                const float *pfFilter,
                const SDimensions3D& dims)
{
	// The filtering is a regular ramp filter per detector line.

	// Generate filter
	// TODO: Check errors
	int iPaddedDetCount = calcNextPowerOfTwo(2 * dims.iProjU);
	int iHalfFFTSize = astra::calcFFTFourierSize(iPaddedDetCount);


	cufftComplex *pHostFilter = new cufftComplex[dims.iProjAngles * iHalfFFTSize];
	memset(pHostFilter, 0, sizeof(cufftComplex) * dims.iProjAngles * iHalfFFTSize);

	if (pfFilter == 0){
		astra::SFilterConfig filter;
		filter.m_eType = astra::FILTER_RAMLAK;
		astraCUDA::genCuFFTFilter(filter, dims.iProjAngles, pHostFilter, iPaddedDetCount, iHalfFFTSize);
	} else {
		for (int i = 0; i < dims.iProjAngles * iHalfFFTSize; i++) {
			pHostFilter[i].x = pfFilter[i];
			pHostFilter[i].y = 0;
		}
	}

	cufftComplex * D_filter;

	astraCUDA::allocateComplexOnDevice(dims.iProjAngles, iHalfFFTSize, &D_filter);
	astraCUDA::uploadComplexArrayToDevice(dims.iProjAngles, iHalfFFTSize, pHostFilter, D_filter);

	delete [] pHostFilter;




	int projPitch = D_projData.pitch/sizeof(float);
	

	// We process one sinogram at a time.
	float* D_sinoData = (float*)D_projData.ptr;

	cufftComplex * D_sinoFFT = NULL;
	astraCUDA::allocateComplexOnDevice(dims.iProjAngles, iHalfFFTSize, &D_sinoFFT);

	bool ok = true;

	for (int v = 0; v < dims.iProjV; ++v) {

		ok = astraCUDA::runCudaFFT(dims.iProjAngles, D_sinoData, projPitch,
		                dims.iProjU, iPaddedDetCount, iHalfFFTSize,
		                D_sinoFFT);

		if (!ok) break;

		astraCUDA::applyFilter(dims.iProjAngles, iHalfFFTSize, D_sinoFFT, D_filter);


		ok = astraCUDA::runCudaIFFT(dims.iProjAngles, D_sinoFFT, D_sinoData, projPitch,
		                 dims.iProjU, iPaddedDetCount, iHalfFFTSize);

		if (!ok) break;

		D_sinoData += (dims.iProjAngles * projPitch);
	}

	astraCUDA::freeComplexOnDevice(D_sinoFFT);
	astraCUDA::freeComplexOnDevice(D_filter);

	return ok;
}


bool FDK(cudaPitchedPtr D_volumeData,
         cudaPitchedPtr D_projData,
         const SConeProjection* angles,
         const SDimensions3D& dims, SProjectorParams3D params, bool bShortScan,
	     const float* pfFilter)
{
	bool ok;

	// NB: We don't support arbitrary cone_vec geometries here.
	// Only those that are vertical sub-geometries
	// (cf. CompositeGeometryManager) of regular cone geometries.
	assert(dims.iProjAngles > 0);
	const SConeProjection& p0 = angles[0];

	// assuming U is in the XY plane, V is parallel to Z axis
	float fDetCX = p0.fDetSX + 0.5*dims.iProjU*p0.fDetUX;
	float fDetCY = p0.fDetSY + 0.5*dims.iProjU*p0.fDetUY;
	float fDetCZ = p0.fDetSZ + 0.5*dims.iProjV*p0.fDetVZ;

	float fSrcOrigin = sqrt(p0.fSrcX*p0.fSrcX + p0.fSrcY*p0.fSrcY);
	float fDetOrigin = sqrt(fDetCX*fDetCX + fDetCY*fDetCY);
	float fDetUSize = sqrt(p0.fDetUX*p0.fDetUX + p0.fDetUY*p0.fDetUY);
	float fDetVSize = abs(p0.fDetVZ);

	float fZShift = fDetCZ - p0.fSrcZ;

	float *pfAngles = new float[dims.iProjAngles];
	for (unsigned int i = 0; i < dims.iProjAngles; ++i) {
		// FIXME: Sign/order
		pfAngles[i] = -atan2(angles[i].fSrcX, angles[i].fSrcY) + M_PI;
	}


#if 1
	// NB: assuming cube voxels (params.fVolScaleX)
	ok = FDK_PreWeight(D_projData, fSrcOrigin, fDetOrigin,
	                fZShift, fDetUSize, fDetVSize, params.fVolScaleX,
	                bShortScan, dims, pfAngles);
#else
	ok = true;
#endif
	delete[] pfAngles;

	if (!ok)
		return false;

#if 1
	// Perform filtering
	ok = FDK_Filter(D_projData, pfFilter, dims);
#endif

	if (!ok)
		return false;

	// Perform BP

	params.bFDKWeighting = true;

	//ok = FDK_BP(D_volumeData, D_projData, fSrcOrigin, fDetOrigin, 0.0f, 0.0f, fDetUSize, fDetVSize, dims, pfAngles);
	ok = ConeBP(D_volumeData, D_projData, dims, angles, params);

	if (!ok)
		return false;

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
