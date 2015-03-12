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

#include "cgls3d.h"
#include "util3d.h"
#include "arith3d.h"
#include "cone_fp.h"

#ifdef STANDALONE
#include "testutil.h"
#endif

namespace astraCUDA3d {

CGLS::CGLS() : ReconAlgo3D()
{
	D_maskData.ptr = 0;
	D_smaskData.ptr = 0;

	D_sinoData.ptr = 0;
	D_volumeData.ptr = 0;

	D_r.ptr = 0;
	D_w.ptr = 0;
	D_z.ptr = 0;
	D_p.ptr = 0;

	useVolumeMask = false;
	useSinogramMask = false;
}


CGLS::~CGLS()
{
	reset();
}

void CGLS::reset()
{
	cudaFree(D_r.ptr);
	cudaFree(D_w.ptr);
	cudaFree(D_z.ptr);
	cudaFree(D_p.ptr);

	D_maskData.ptr = 0;
	D_smaskData.ptr = 0;

	D_sinoData.ptr = 0;
	D_volumeData.ptr = 0;

	D_r.ptr = 0;
	D_w.ptr = 0;
	D_z.ptr = 0;
	D_p.ptr = 0;

	useVolumeMask = false;
	useSinogramMask = false;

	sliceInitialized = false;

	ReconAlgo3D::reset();
}

bool CGLS::enableVolumeMask()
{
	useVolumeMask = true;
	return true;
}

bool CGLS::enableSinogramMask()
{
	useSinogramMask = true;
	return true;
}


bool CGLS::init()
{
	D_z = allocateVolumeData(dims);
	D_p = allocateVolumeData(dims);
	D_r = allocateProjectionData(dims);
	D_w = allocateProjectionData(dims);

	// TODO: check if allocations succeeded
	return true;
}

bool CGLS::setVolumeMask(cudaPitchedPtr& _D_maskData)
{
	assert(useVolumeMask);

	D_maskData = _D_maskData;

	return true;
}

bool CGLS::setSinogramMask(cudaPitchedPtr& _D_smaskData)
{
	return false;
#if 0
	// TODO: Implement this
	assert(useSinogramMask);

	D_smaskData = _D_smaskData;
	return true;
#endif
}

bool CGLS::setBuffers(cudaPitchedPtr& _D_volumeData,
                      cudaPitchedPtr& _D_projData)
{
	D_volumeData = _D_volumeData;
	D_sinoData = _D_projData;

	sliceInitialized = false;

	return true;
}

bool CGLS::iterate(unsigned int iterations)
{
	shouldAbort = false;

	if (!sliceInitialized) {

		// copy sinogram
		duplicateProjectionData(D_r, D_sinoData, dims);

		// r = sino - A*x
		if (useVolumeMask) {
				duplicateVolumeData(D_z, D_volumeData, dims);
				processVol3D<opMul>(D_z, D_maskData, dims);
				callFP(D_z, D_r, -1.0f);
		} else {
				callFP(D_volumeData, D_r, -1.0f);
		}

		// p = A'*r
		zeroVolumeData(D_p, dims);
		callBP(D_p, D_r, 1.0f);
		if (useVolumeMask)
			processVol3D<opMul>(D_p, D_maskData, dims);

		gamma = dotProduct3D(D_p, dims.iVolX, dims.iVolY, dims.iVolZ);

		sliceInitialized = true;

	}


	// iteration
	for (unsigned int iter = 0; iter < iterations && !shouldAbort; ++iter) {

		// w = A*p
		zeroProjectionData(D_w, dims);
		callFP(D_p, D_w, 1.0f);

		// alpha = gamma / <w,w>
		float ww = dotProduct3D(D_w, dims.iProjU, dims.iProjAngles, dims.iProjV);
		float alpha = gamma / ww;

		// x += alpha*p
		processVol3D<opAddScaled>(D_volumeData, D_p, alpha, dims);

		// r -= alpha*w
		processSino3D<opAddScaled>(D_r, D_w, -alpha, dims);

		// z = A'*r
		zeroVolumeData(D_z, dims);
		callBP(D_z, D_r, 1.0f);
		if (useVolumeMask)
			processVol3D<opMul>(D_z, D_maskData, dims);

		float beta = 1.0f / gamma;
		gamma = dotProduct3D(D_z, dims.iVolX, dims.iVolY, dims.iVolZ);

		beta *= gamma;

		// p = z + beta*p
		processVol3D<opScaleAndAdd>(D_p, D_z, beta, dims);
	}

	return true;
}

float CGLS::computeDiffNorm()
{
	// We can use w and z as temporary storage here since they're not
	// used outside of iterations.

	// copy sinogram to w
	duplicateProjectionData(D_w, D_sinoData, dims);

	// do FP, subtracting projection from sinogram
	if (useVolumeMask) {
			duplicateVolumeData(D_z, D_volumeData, dims);
			processVol3D<opMul>(D_z, D_maskData, dims);
			callFP(D_z, D_w, -1.0f);
	} else {
			callFP(D_volumeData, D_w, -1.0f);
	}

	float s = dotProduct3D(D_w, dims.iProjU, dims.iProjAngles, dims.iProjV);
	return sqrt(s);
}


bool doCGLS(cudaPitchedPtr& D_volumeData, 
            cudaPitchedPtr& D_sinoData,
            cudaPitchedPtr& D_maskData,
            const SDimensions3D& dims, const SConeProjection* angles,
            unsigned int iterations)
{
	CGLS cgls;
	bool ok = true;

	ok &= cgls.setConeGeometry(dims, angles, 1.0f);
	if (D_maskData.ptr)
		ok &= cgls.enableVolumeMask();

	if (!ok)
		return false;

	ok = cgls.init();
	if (!ok)
		return false;

	if (D_maskData.ptr)
		ok &= cgls.setVolumeMask(D_maskData);

	ok &= cgls.setBuffers(D_volumeData, D_sinoData);
	if (!ok)
		return false;

	ok = cgls.iterate(iterations);

	return ok;
}

}

#ifdef STANDALONE

using namespace astraCUDA3d;

int main()
{
	SDimensions3D dims;
	dims.iVolX = 256;
	dims.iVolY = 256;
	dims.iVolZ = 256;
	dims.iProjAngles = 100;
	dims.iProjU = 512;
	dims.iProjV = 512;
	dims.iRaysPerDet = 1;

	SConeProjection angle[100];
	angle[0].fSrcX = -2905.6;
	angle[0].fSrcY = 0;
	angle[0].fSrcZ = 0;

	angle[0].fDetSX = 694.4;
	angle[0].fDetSY = -122.4704;
	angle[0].fDetSZ = -122.4704;

	angle[0].fDetUX = 0;
	angle[0].fDetUY = .4784;
	//angle[0].fDetUY = .5;
	angle[0].fDetUZ = 0;

	angle[0].fDetVX = 0;
	angle[0].fDetVY = 0;
	angle[0].fDetVZ = .4784;

#define ROTATE0(name,i,alpha) do { angle[i].f##name##X = angle[0].f##name##X * cos(alpha) - angle[0].f##name##Y * sin(alpha); angle[i].f##name##Y = angle[0].f##name##X * sin(alpha) + angle[0].f##name##Y * cos(alpha); } while(0)
	for (int i = 1; i < 100; ++i) {
		angle[i] = angle[0];
		ROTATE0(Src, i, i*2*M_PI/100);
		ROTATE0(DetS, i, i*2*M_PI/100);
		ROTATE0(DetU, i, i*2*M_PI/100);
		ROTATE0(DetV, i, i*2*M_PI/100);
	}
#undef ROTATE0


	cudaPitchedPtr volData = allocateVolumeData(dims);
	cudaPitchedPtr projData = allocateProjectionData(dims);
	zeroProjectionData(projData, dims);

	float* pbuf = new float[100*512*512];
	copyProjectionsFromDevice(pbuf, projData, dims);
	copyProjectionsToDevice(pbuf, projData, dims);
	delete[] pbuf;

#if 0
	float* slice = new float[256*256];
	cudaPitchedPtr ptr;
	ptr.ptr = slice;
	ptr.pitch = 256*sizeof(float);
	ptr.xsize = 256*sizeof(float);
	ptr.ysize = 256;

	for (unsigned int i = 0; i < 256; ++i) {
		for (unsigned int y = 0; y < 256; ++y)
			for (unsigned int x = 0; x < 256; ++x)
				slice[y*256+x] = (i-127.5)*(i-127.5)+(y-127.5)*(y-127.5)+(x-127.5)*(x-127.5) < 4900 ? 1.0f : 0.0f;

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
	}
	astraCUDA3d::ConeFP(volData, projData, dims, angle, 1.0f);

#else

	for (int i = 0; i < 100; ++i) {
		char fname[32];
		sprintf(fname, "Tiffs/%04d.png", 4*i);
		unsigned int w,h;
		float* bufp = loadImage(fname, w,h);

		for (int j = 0; j < 512*512; ++j) {
			float v = bufp[j];
			if (v > 236.0f) v = 236.0f;
			v = logf(236.0f / v);
			bufp[j] = 256*v;
		}

		for (int j = 0; j < 512; ++j) {
			cudaMemcpy(((float*)projData.ptr)+100*512*j+512*i, bufp+512*j, 512*sizeof(float), cudaMemcpyHostToDevice);
		}

		delete[] bufp;

	}
#endif

#if 0
	float* bufs = new float[100*512];

	for (int i = 0; i < 512; ++i) {
		cudaMemcpy(bufs, ((float*)projData.ptr)+100*512*i, 100*512*sizeof(float), cudaMemcpyDeviceToHost);

		printf("%d %d %d\n", projData.pitch, projData.xsize, projData.ysize);

		char fname[20];
		sprintf(fname, "sino%03d.png", i);
		saveImage(fname, 100, 512, bufs);
	}

	float* bufp = new float[512*512];

	for (int i = 0; i < 100; ++i) {
		for (int j = 0; j < 512; ++j) {
			cudaMemcpy(bufp+512*j, ((float*)projData.ptr)+100*512*j+512*i, 512*sizeof(float), cudaMemcpyDeviceToHost);
		}

		char fname[20];
		sprintf(fname, "proj%03d.png", i);
		saveImage(fname, 512, 512, bufp);
	}
#endif

	zeroVolumeData(volData, dims);

	cudaPitchedPtr maskData;
	maskData.ptr = 0;

	astraCUDA3d::doCGLS(volData, projData, maskData, dims, angle, 50);
#if 1
	float* buf = new float[256*256];

	for (int i = 0; i < 256; ++i) {
		cudaMemcpy(buf, ((float*)volData.ptr)+256*256*i, 256*256*sizeof(float), cudaMemcpyDeviceToHost);

		char fname[20];
		sprintf(fname, "vol%03d.png", i);
		saveImage(fname, 256, 256, buf);
	}
#endif

	return 0;
}
#endif

