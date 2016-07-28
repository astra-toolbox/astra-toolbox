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

#include "sirt3d.h"
#include "util3d.h"
#include "arith3d.h"
#include "cone_fp.h"

#ifdef STANDALONE
#include "testutil.h"
#endif

namespace astraCUDA3d {

SIRT::SIRT() : ReconAlgo3D()
{
	D_maskData.ptr = 0;
	D_smaskData.ptr = 0;

	D_sinoData.ptr = 0;
	D_volumeData.ptr = 0;

	D_projData.ptr = 0;
	D_tmpData.ptr = 0;

	D_lineWeight.ptr = 0;
	D_pixelWeight.ptr = 0;

	useVolumeMask = false;
	useSinogramMask = false;

	useMinConstraint = false;
	useMaxConstraint = false;

	fRelaxation = 1.0f;
}


SIRT::~SIRT()
{
	reset();
}

void SIRT::reset()
{
	cudaFree(D_projData.ptr);
	cudaFree(D_tmpData.ptr);
	cudaFree(D_lineWeight.ptr);
	cudaFree(D_pixelWeight.ptr);

	D_maskData.ptr = 0;
	D_smaskData.ptr = 0;

	D_sinoData.ptr = 0;
	D_volumeData.ptr = 0;

	D_projData.ptr = 0;
	D_tmpData.ptr = 0;

	D_lineWeight.ptr = 0;
	D_pixelWeight.ptr = 0;

	useVolumeMask = false;
	useSinogramMask = false;

	fRelaxation = 1.0f;

	ReconAlgo3D::reset();
}

bool SIRT::enableVolumeMask()
{
	useVolumeMask = true;
	return true;
}

bool SIRT::enableSinogramMask()
{
	useSinogramMask = true;
	return true;
}


bool SIRT::init()
{
	D_pixelWeight = allocateVolumeData(dims);
	zeroVolumeData(D_pixelWeight, dims);

	D_tmpData = allocateVolumeData(dims);
	zeroVolumeData(D_tmpData, dims);

	D_projData = allocateProjectionData(dims);
	zeroProjectionData(D_projData, dims);

	D_lineWeight = allocateProjectionData(dims);
	zeroProjectionData(D_lineWeight, dims);

	// We can't precompute lineWeights and pixelWeights when using a mask
	if (!useVolumeMask && !useSinogramMask)
		precomputeWeights();

	// TODO: check if allocations succeeded
	return true;
}

bool SIRT::setMinConstraint(float fMin)
{
	fMinConstraint = fMin;
	useMinConstraint = true;
	return true;
}

bool SIRT::setMaxConstraint(float fMax)
{
	fMaxConstraint = fMax;
	useMaxConstraint = true;
	return true;
}

bool SIRT::precomputeWeights()
{
	zeroProjectionData(D_lineWeight, dims);
	if (useVolumeMask) {
		callFP(D_maskData, D_lineWeight, 1.0f);
	} else {
		processVol3D<opSet>(D_tmpData, 1.0f, dims);
		callFP(D_tmpData, D_lineWeight, 1.0f);
	}
	processSino3D<opInvert>(D_lineWeight, dims);

	if (useSinogramMask) {
		// scale line weights with sinogram mask to zero out masked sinogram pixels
		processSino3D<opMul>(D_lineWeight, D_smaskData, dims);
	}

	zeroVolumeData(D_pixelWeight, dims);

	if (useSinogramMask) {
		callBP(D_pixelWeight, D_smaskData, 1.0f);
	} else {
		processSino3D<opSet>(D_projData, 1.0f, dims);
		callBP(D_pixelWeight, D_projData, 1.0f);
	}
#if 0
	float* bufp = new float[512*512];

	for (int i = 0; i < 180; ++i) {
		for (int j = 0; j < 512; ++j) {
			cudaMemcpy(bufp+512*j, ((float*)D_projData.ptr)+180*512*j+512*i, 512*sizeof(float), cudaMemcpyDeviceToHost);
		}

		char fname[20];
		sprintf(fname, "ray%03d.png", i);
		saveImage(fname, 512, 512, bufp);
	}
#endif

#if 0
	float* buf = new float[256*256];

	for (int i = 0; i < 256; ++i) {
		cudaMemcpy(buf, ((float*)D_pixelWeight.ptr)+256*256*i, 256*256*sizeof(float), cudaMemcpyDeviceToHost);

		char fname[20];
		sprintf(fname, "pix%03d.png", i);
		saveImage(fname, 256, 256, buf);
	}
#endif
	processVol3D<opInvert>(D_pixelWeight, dims);

	if (useVolumeMask) {
		// scale pixel weights with mask to zero out masked pixels
		processVol3D<opMul>(D_pixelWeight, D_maskData, dims);
	}
	processVol3D<opMul>(D_pixelWeight, fRelaxation, dims);


	return true;
}


bool SIRT::setVolumeMask(cudaPitchedPtr& _D_maskData)
{
	assert(useVolumeMask);

	D_maskData = _D_maskData;

	return true;
}

bool SIRT::setSinogramMask(cudaPitchedPtr& _D_smaskData)
{
	assert(useSinogramMask);

	D_smaskData = _D_smaskData;

	return true;
}

bool SIRT::setBuffers(cudaPitchedPtr& _D_volumeData,
                      cudaPitchedPtr& _D_projData)
{
	D_volumeData = _D_volumeData;
	D_sinoData = _D_projData;

	return true;
}

bool SIRT::iterate(unsigned int iterations)
{
	shouldAbort = false;

	if (useVolumeMask || useSinogramMask)
		precomputeWeights();

#if 0
	float* buf = new float[256*256];

	for (int i = 0; i < 256; ++i) {
		cudaMemcpy(buf, ((float*)D_pixelWeight.ptr)+256*256*i, 256*256*sizeof(float), cudaMemcpyDeviceToHost);

		char fname[20];
		sprintf(fname, "pix%03d.png", i);
		saveImage(fname, 256, 256, buf);
	}
#endif
#if 0
	float* bufp = new float[512*512];

	for (int i = 0; i < 100; ++i) {
		for (int j = 0; j < 512; ++j) {
			cudaMemcpy(bufp+512*j, ((float*)D_lineWeight.ptr)+100*512*j+512*i, 512*sizeof(float), cudaMemcpyDeviceToHost);
		}

		char fname[20];
		sprintf(fname, "ray%03d.png", i);
		saveImage(fname, 512, 512, bufp);
	}
#endif


	// iteration
	for (unsigned int iter = 0; iter < iterations && !shouldAbort; ++iter) {
		// copy sinogram to projection data
		duplicateProjectionData(D_projData, D_sinoData, dims);

		// do FP, subtracting projection from sinogram
		if (useVolumeMask) {
				duplicateVolumeData(D_tmpData, D_volumeData, dims);
				processVol3D<opMul>(D_tmpData, D_maskData, dims);
				callFP(D_tmpData, D_projData, -1.0f);
		} else {
				callFP(D_volumeData, D_projData, -1.0f);
		}

		processSino3D<opMul>(D_projData, D_lineWeight, dims);

		zeroVolumeData(D_tmpData, dims);
#if 0
	float* bufp = new float[512*512];
	printf("Dumping projData: %p\n", (void*)D_projData.ptr);
	for (int i = 0; i < 180; ++i) {
		for (int j = 0; j < 512; ++j) {
			cudaMemcpy(bufp+512*j, ((float*)D_projData.ptr)+180*512*j+512*i, 512*sizeof(float), cudaMemcpyDeviceToHost);
		}

		char fname[20];
		sprintf(fname, "diff%03d.png", i);
		saveImage(fname, 512, 512, bufp);
	}
#endif


		callBP(D_tmpData, D_projData, 1.0f);
#if 0
	printf("Dumping tmpData: %p\n", (void*)D_tmpData.ptr);
	float* buf = new float[256*256];

	for (int i = 0; i < 256; ++i) {
		cudaMemcpy(buf, ((float*)D_tmpData.ptr)+256*256*i, 256*256*sizeof(float), cudaMemcpyDeviceToHost);

		char fname[20];
		sprintf(fname, "add%03d.png", i);
		saveImage(fname, 256, 256, buf);
	}
#endif

		// pixel weights also contain the volume mask and relaxation factor
		processVol3D<opAddMul>(D_volumeData, D_tmpData, D_pixelWeight, dims);

		if (useMinConstraint)
			processVol3D<opClampMin>(D_volumeData, fMinConstraint, dims);
		if (useMaxConstraint)
			processVol3D<opClampMax>(D_volumeData, fMaxConstraint, dims);
	}

	return true;
}

float SIRT::computeDiffNorm()
{
	// copy sinogram to projection data
	duplicateProjectionData(D_projData, D_sinoData, dims);

	// do FP, subtracting projection from sinogram
	if (useVolumeMask) {
			duplicateVolumeData(D_tmpData, D_volumeData, dims);
			processVol3D<opMul>(D_tmpData, D_maskData, dims);
			callFP(D_tmpData, D_projData, -1.0f);
	} else {
			callFP(D_volumeData, D_projData, -1.0f);
	}

	float s = dotProduct3D(D_projData, dims.iProjU, dims.iProjAngles, dims.iProjV);
	return sqrt(s);
}


bool doSIRT(cudaPitchedPtr& D_volumeData, 
            cudaPitchedPtr& D_sinoData,
            cudaPitchedPtr& D_maskData,
            const SDimensions3D& dims, const SConeProjection* angles,
            unsigned int iterations)
{
	SIRT sirt;
	bool ok = true;

	ok &= sirt.setConeGeometry(dims, angles, 1.0f);
	if (D_maskData.ptr)
		ok &= sirt.enableVolumeMask();

	if (!ok)
		return false;

	ok = sirt.init();
	if (!ok)
		return false;

	if (D_maskData.ptr)
		ok &= sirt.setVolumeMask(D_maskData);

	ok &= sirt.setBuffers(D_volumeData, D_sinoData);
	if (!ok)
		return false;

	ok = sirt.iterate(iterations);

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

	astraCUDA3d::doSIRT(volData, projData, maskData, dims, angle, 50);
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

