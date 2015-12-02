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

#include "cgls.h"
#include "util.h"
#include "arith.h"

#ifdef STANDALONE
#include "testutil.h"
#endif

namespace astraCUDA {

CGLS::CGLS() : ReconAlgo()
{
	D_z = 0;
	D_p = 0;
	D_r = 0;
	D_w = 0;

	sliceInitialized = false;
}


CGLS::~CGLS()
{
	reset();
}

void CGLS::reset()
{
	cudaFree(D_z);
	cudaFree(D_p);
	cudaFree(D_r);
	cudaFree(D_w);

	D_z = 0;
	D_p = 0;
	D_r = 0;
	D_w = 0;

	ReconAlgo::reset();
}

bool CGLS::init()
{
	// Lifetime of z: within an iteration
	allocateVolumeData(D_z, zPitch, dims);

	// Lifetime of p: full algorithm
	allocateVolumeData(D_p, pPitch, dims);

	// Lifetime of r: full algorithm
	allocateProjectionData(D_r, rPitch, dims);
	
	// Lifetime of w: within an iteration
	allocateProjectionData(D_w, wPitch, dims);

	// TODO: check if allocations succeeded
	return true;
}


bool CGLS::setBuffers(float* _D_volumeData, unsigned int _volumePitch,
                      float* _D_projData, unsigned int _projPitch)
{
	bool ok = ReconAlgo::setBuffers(_D_volumeData, _volumePitch,
	                                _D_projData, _projPitch);

	if (!ok)
		return false;

	sliceInitialized = false;

	return true;
}

bool CGLS::copyDataToGPU(const float* pfSinogram, unsigned int iSinogramPitch, float fSinogramScale,
                         const float* pfReconstruction, unsigned int iReconstructionPitch,
                         const float* pfVolMask, unsigned int iVolMaskPitch,
                         const float* pfSinoMask, unsigned int iSinoMaskPitch)
{
	sliceInitialized = false;

	return ReconAlgo::copyDataToGPU(pfSinogram, iSinogramPitch, fSinogramScale, pfReconstruction, iReconstructionPitch, pfVolMask, iVolMaskPitch, pfSinoMask, iSinoMaskPitch);
}

bool CGLS::iterate(unsigned int iterations)
{
	shouldAbort = false;

	if (!sliceInitialized) {

		// copy sinogram
		duplicateProjectionData(D_r, D_sinoData, sinoPitch, dims);

		// r = sino - A*x
		if (useVolumeMask) {
			// Use z as temporary storage here since it is unused
			duplicateVolumeData(D_z, D_volumeData, volumePitch, dims);
			processVol<opMul>(D_z, D_maskData, zPitch, dims);
			callFP(D_z, zPitch, D_r, rPitch, -1.0f);
		} else {
			callFP(D_volumeData, volumePitch, D_r, rPitch, -1.0f);
		}


		// p = A'*r
		zeroVolumeData(D_p, pPitch, dims);
		callBP(D_p, pPitch, D_r, rPitch, 1.0f);
		if (useVolumeMask)
			processVol<opMul>(D_p, D_maskData, pPitch, dims);


		gamma = dotProduct2D(D_p, pPitch, dims.iVolWidth, dims.iVolHeight);

		sliceInitialized = true;
	}


	// iteration
	for (unsigned int iter = 0; iter < iterations && !shouldAbort; ++iter) {

		// w = A*p
		zeroProjectionData(D_w, wPitch, dims);
		callFP(D_p, pPitch, D_w, wPitch, 1.0f);

		// alpha = gamma / <w,w>
		float ww = dotProduct2D(D_w, wPitch, dims.iProjDets, dims.iProjAngles);
		float alpha = gamma / ww;

		// x += alpha*p
		processVol<opAddScaled>(D_volumeData, D_p, alpha, volumePitch, dims);

		// r -= alpha*w
		processSino<opAddScaled>(D_r, D_w, -alpha, rPitch, dims);


		// z = A'*r
		zeroVolumeData(D_z, zPitch, dims);
		callBP(D_z, zPitch, D_r, rPitch, 1.0f);
		if (useVolumeMask)
			processVol<opMul>(D_z, D_maskData, zPitch, dims);

		float beta = 1.0f / gamma;
		gamma = dotProduct2D(D_z, zPitch, dims.iVolWidth, dims.iVolHeight);
		beta *= gamma;

		// p = z + beta*p
		processVol<opScaleAndAdd>(D_p, D_z, beta, pPitch, dims);

	}

	return true;
}


float CGLS::computeDiffNorm()
{
	// We can use w and z as temporary storage here since they're not
	// used outside of iterations.

	// copy sinogram to w
	duplicateProjectionData(D_w, D_sinoData, sinoPitch, dims);

	// do FP, subtracting projection from sinogram
	if (useVolumeMask) {
			duplicateVolumeData(D_z, D_volumeData, volumePitch, dims);
			processVol<opMul>(D_z, D_maskData, zPitch, dims);
			callFP(D_z, zPitch, D_w, wPitch, -1.0f);
	} else {
			callFP(D_volumeData, volumePitch, D_w, wPitch, -1.0f);
	}

	// compute norm of D_w

	float s = dotProduct2D(D_w, wPitch, dims.iProjDets, dims.iProjAngles);

	return sqrt(s);
}

bool doCGLS(float* D_volumeData, unsigned int volumePitch,
            float* D_sinoData, unsigned int sinoPitch,
            const SDimensions& dims, /*const SAugmentedData& augs,*/
            const float* angles, const float* TOffsets, unsigned int iterations)
{
	CGLS cgls;
	bool ok = true;

	ok &= cgls.setGeometry(dims, angles);
#if 0
	if (D_maskData)
		ok &= cgls.enableVolumeMask();
#endif
	if (TOffsets)
		ok &= cgls.setTOffsets(TOffsets);

	if (!ok)
		return false;

	ok = cgls.init();
	if (!ok)
		return false;

#if 0
	if (D_maskData)
		ok &= cgls.setVolumeMask(D_maskData, maskPitch);
#endif

	ok &= cgls.setBuffers(D_volumeData, volumePitch, D_sinoData, sinoPitch);
	if (!ok)
		return false;

	ok = cgls.iterate(iterations);

	return ok;
}

}

#ifdef STANDALONE

using namespace astraCUDA;

int main()
{
	float* D_volumeData;
	float* D_sinoData;

	SDimensions dims;
	dims.iVolWidth = 1024;
	dims.iVolHeight = 1024;
	dims.iProjAngles = 512;
	dims.iProjDets = 1536;
	dims.fDetScale = 1.0f;
	dims.iRaysPerDet = 1;
	unsigned int volumePitch, sinoPitch;

	allocateVolume(D_volumeData, dims.iVolWidth, dims.iVolHeight, volumePitch);
	zeroVolume(D_volumeData, volumePitch, dims.iVolWidth, dims.iVolHeight);
	printf("pitch: %u\n", volumePitch);

	allocateVolume(D_sinoData, dims.iProjDets, dims.iProjAngles, sinoPitch);
	zeroVolume(D_sinoData, sinoPitch, dims.iProjDets, dims.iProjAngles);
	printf("pitch: %u\n", sinoPitch);
	
	unsigned int y, x;
	float* sino = loadImage("sino.png", y, x);

	float* img = new float[dims.iVolWidth*dims.iVolHeight];

	copySinogramToDevice(sino, dims.iProjDets, dims.iProjDets, dims.iProjAngles, D_sinoData, sinoPitch);

	float* angle = new float[dims.iProjAngles];

	for (unsigned int i = 0; i < dims.iProjAngles; ++i)
		angle[i] = i*(M_PI/dims.iProjAngles);

	CGLS cgls;

	cgls.setGeometry(dims, angle);
	cgls.init();

	cgls.setBuffers(D_volumeData, volumePitch, D_sinoData, sinoPitch);

	cgls.iterate(25);

	delete[] angle;

	copyVolumeFromDevice(img, dims.iVolWidth, dims.iVolWidth, dims.iVolHeight, D_volumeData, volumePitch);

	saveImage("vol.png",dims.iVolHeight,dims.iVolWidth,img);

	return 0;
}
#endif
