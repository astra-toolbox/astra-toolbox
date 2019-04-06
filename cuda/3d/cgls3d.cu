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

#include "astra/cuda/3d/cgls3d.h"
#include "astra/cuda/3d/util3d.h"
#include "astra/cuda/3d/arith3d.h"
#include "astra/cuda/3d/cone_fp.h"

#include <cstdio>
#include <cassert>

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
	for (unsigned int iter = 0; iter < iterations && !astra::shouldAbort(); ++iter) {

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

	ok &= cgls.setConeGeometry(dims, angles, SProjectorParams3D());
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
