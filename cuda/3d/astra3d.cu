/*
-----------------------------------------------------------------------
Copyright 2012 iMinds-Vision Lab, University of Antwerp

Contact: astra@ua.ac.be
Website: http://astra.ua.ac.be


This file is part of the
All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA Toolbox").

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
#include "sirt3d.h"
#include "util3d.h"
#include "cone_fp.h"
#include "cone_bp.h"
#include "par3d_fp.h"
#include "par3d_bp.h"
#include "fdk.h"
#include "arith3d.h"
#include "astra3d.h"

#include <iostream>

using namespace astraCUDA3d;

namespace astra {

enum CUDAProjectionType3d {
	PROJ_PARALLEL,
	PROJ_CONE
};


static SConeProjection* genConeProjections(unsigned int iProjAngles,
                                           unsigned int iProjU,
                                           unsigned int iProjV,
                                           double fOriginSourceDistance,
                                           double fOriginDetectorDistance,
                                           double fDetUSize,
                                           double fDetVSize,
                                           const float *pfAngles)
{
	SConeProjection base;
	base.fSrcX = 0.0f;
	base.fSrcY = -fOriginSourceDistance;
	base.fSrcZ = 0.0f;

	base.fDetSX = iProjU * fDetUSize * -0.5f;
	base.fDetSY = fOriginDetectorDistance;
	base.fDetSZ = iProjV * fDetVSize * -0.5f;

	base.fDetUX = fDetUSize;
	base.fDetUY = 0.0f;
	base.fDetUZ = 0.0f;

	base.fDetVX = 0.0f;
	base.fDetVY = 0.0f;
	base.fDetVZ = fDetVSize;

	SConeProjection* p = new SConeProjection[iProjAngles];

#define ROTATE0(name,i,alpha) do { p[i].f##name##X = base.f##name##X * cos(alpha) - base.f##name##Y * sin(alpha); p[i].f##name##Y = base.f##name##X * sin(alpha) + base.f##name##Y * cos(alpha); p[i].f##name##Z = base.f##name##Z; } while(0)

	for (unsigned int i = 0; i < iProjAngles; ++i) {
		ROTATE0(Src, i, pfAngles[i]);
		ROTATE0(DetS, i, pfAngles[i]);
		ROTATE0(DetU, i, pfAngles[i]);
		ROTATE0(DetV, i, pfAngles[i]);
	}

#undef ROTATE0

	return p;
}

static SPar3DProjection* genPar3DProjections(unsigned int iProjAngles,
                                             unsigned int iProjU,
                                             unsigned int iProjV,
                                             double fDetUSize,
                                             double fDetVSize,
                                             const float *pfAngles)
{
	SPar3DProjection base;
	base.fRayX = 0.0f;
	base.fRayY = 1.0f;
	base.fRayZ = 0.0f;

	base.fDetSX = iProjU * fDetUSize * -0.5f;
	base.fDetSY = 0.0f;
	base.fDetSZ = iProjV * fDetVSize * -0.5f;

	base.fDetUX = fDetUSize;
	base.fDetUY = 0.0f;
	base.fDetUZ = 0.0f;

	base.fDetVX = 0.0f;
	base.fDetVY = 0.0f;
	base.fDetVZ = fDetVSize;

	SPar3DProjection* p = new SPar3DProjection[iProjAngles];

#define ROTATE0(name,i,alpha) do { p[i].f##name##X = base.f##name##X * cos(alpha) - base.f##name##Y * sin(alpha); p[i].f##name##Y = base.f##name##X * sin(alpha) + base.f##name##Y * cos(alpha); p[i].f##name##Z = base.f##name##Z; } while(0)

	for (unsigned int i = 0; i < iProjAngles; ++i) {
		ROTATE0(Ray, i, pfAngles[i]);
		ROTATE0(DetS, i, pfAngles[i]);
		ROTATE0(DetU, i, pfAngles[i]);
		ROTATE0(DetV, i, pfAngles[i]);
	}

#undef ROTATE0

	return p;
}




class AstraSIRT3d_internal {
public:
	SDimensions3D dims;
	CUDAProjectionType3d projType;

	float* angles;
	float fOriginSourceDistance;
	float fOriginDetectorDistance;
	float fSourceZ;
	float fDetSize;

	SConeProjection* projs;
	SPar3DProjection* parprojs;

	float fPixelSize;

	bool initialized;
	bool setStartReconstruction;

	bool useVolumeMask;
	bool useSinogramMask;

	// Input/output
	cudaPitchedPtr D_projData;
	cudaPitchedPtr D_volumeData;
	cudaPitchedPtr D_maskData;
	cudaPitchedPtr D_smaskData;

	SIRT sirt;
};

AstraSIRT3d::AstraSIRT3d()
{
	pData = new AstraSIRT3d_internal();

	pData->angles = 0;
	pData->D_projData.ptr = 0;
	pData->D_volumeData.ptr = 0;
	pData->D_maskData.ptr = 0;
	pData->D_smaskData.ptr = 0;

	pData->dims.iVolX = 0;
	pData->dims.iVolY = 0;
	pData->dims.iVolZ = 0;
	pData->dims.iProjAngles = 0;
	pData->dims.iProjU = 0;
	pData->dims.iProjV = 0;
	pData->dims.iRaysPerDetDim = 1;
	pData->dims.iRaysPerVoxelDim = 1;

	pData->projs = 0;

	pData->initialized = false;
	pData->setStartReconstruction = false;

	pData->useVolumeMask = false;
	pData->useSinogramMask = false;
}

AstraSIRT3d::~AstraSIRT3d()
{
	delete[] pData->angles;
	pData->angles = 0;

	delete[] pData->projs;
	pData->projs = 0;

	cudaFree(pData->D_projData.ptr);
	pData->D_projData.ptr = 0;

	cudaFree(pData->D_volumeData.ptr);
	pData->D_volumeData.ptr = 0;

	cudaFree(pData->D_maskData.ptr);
	pData->D_maskData.ptr = 0;

	cudaFree(pData->D_smaskData.ptr);
	pData->D_smaskData.ptr = 0;

	delete pData;
	pData = 0;
}

bool AstraSIRT3d::setReconstructionGeometry(unsigned int iVolX,
                                            unsigned int iVolY,
                                            unsigned int iVolZ/*,
                                            float fPixelSize = 1.0f*/)
{
	if (pData->initialized)
		return false;

	pData->dims.iVolX = iVolX;
	pData->dims.iVolY = iVolY;
	pData->dims.iVolZ = iVolZ;

	return (iVolX > 0 && iVolY > 0 && iVolZ > 0);
}


bool AstraSIRT3d::setPar3DGeometry(unsigned int iProjAngles,
                                   unsigned int iProjU,
                                   unsigned int iProjV,
                                   const SPar3DProjection* projs)
{
	if (pData->initialized)
		return false;

	pData->dims.iProjAngles = iProjAngles;
	pData->dims.iProjU = iProjU;
	pData->dims.iProjV = iProjV;

	if (iProjAngles == 0 || iProjU == 0 || iProjV == 0 || projs == 0)
		return false;

	pData->parprojs = new SPar3DProjection[iProjAngles];
	memcpy(pData->parprojs, projs, iProjAngles * sizeof(projs[0]));

	pData->projType = PROJ_PARALLEL;

	return true;
}

bool AstraSIRT3d::setPar3DGeometry(unsigned int iProjAngles,
                                   unsigned int iProjU,
                                   unsigned int iProjV,
                                   float fDetUSize,
                                   float fDetVSize,
                                   const float *pfAngles)
{
	if (pData->initialized)
		return false;

	if (iProjAngles == 0 || iProjU == 0 || iProjV == 0 || pfAngles == 0)
		return false;

	SPar3DProjection* p = genPar3DProjections(iProjAngles,
                                              iProjU, iProjV,
                                              fDetUSize, fDetVSize,
                                              pfAngles);
	pData->dims.iProjAngles = iProjAngles;
	pData->dims.iProjU = iProjU;
	pData->dims.iProjV = iProjV;

	pData->parprojs = p;
	pData->projType = PROJ_PARALLEL;

	return true;
}



bool AstraSIRT3d::setConeGeometry(unsigned int iProjAngles,
                                  unsigned int iProjU,
                                  unsigned int iProjV,
                                  const SConeProjection* projs)
{
	if (pData->initialized)
		return false;

	pData->dims.iProjAngles = iProjAngles;
	pData->dims.iProjU = iProjU;
	pData->dims.iProjV = iProjV;

	if (iProjAngles == 0 || iProjU == 0 || iProjV == 0 || projs == 0)
		return false;

	pData->projs = new SConeProjection[iProjAngles];
	memcpy(pData->projs, projs, iProjAngles * sizeof(projs[0]));

	pData->projType = PROJ_CONE;

	return true;
}

bool AstraSIRT3d::setConeGeometry(unsigned int iProjAngles,
                                  unsigned int iProjU,
                                  unsigned int iProjV,
                                  float fOriginSourceDistance,
                                  float fOriginDetectorDistance,
                                  float fDetUSize,
                                  float fDetVSize,
                                  const float *pfAngles)
{
	if (pData->initialized)
		return false;

	if (iProjAngles == 0 || iProjU == 0 || iProjV == 0 || pfAngles == 0)
		return false;

	SConeProjection* p = genConeProjections(iProjAngles,
                                            iProjU, iProjV,
                                            fOriginSourceDistance,
                                            fOriginDetectorDistance,
                                            fDetUSize, fDetVSize,
                                            pfAngles);
	pData->dims.iProjAngles = iProjAngles;
	pData->dims.iProjU = iProjU;
	pData->dims.iProjV = iProjV;

	pData->projs = p;
	pData->projType = PROJ_CONE;

	return true;
}

bool AstraSIRT3d::enableSuperSampling(unsigned int iVoxelSuperSampling,
                                      unsigned int iDetectorSuperSampling)
{
	if (pData->initialized)
		return false;

	if (iVoxelSuperSampling == 0 || iDetectorSuperSampling == 0)
		return false;

	pData->dims.iRaysPerVoxelDim = iVoxelSuperSampling;
	pData->dims.iRaysPerDetDim = iDetectorSuperSampling;

	return true;
}

bool AstraSIRT3d::enableVolumeMask()
{
	if (pData->initialized)
		return false;

	bool ok = pData->sirt.enableVolumeMask();
	pData->useVolumeMask = ok;

	return ok;
}

bool AstraSIRT3d::enableSinogramMask()
{
	if (pData->initialized)
		return false;

	bool ok = pData->sirt.enableSinogramMask();
	pData->useSinogramMask = ok;

	return ok;
}
	
bool AstraSIRT3d::setGPUIndex(int index)
{
	if (index != -1) {
		cudaSetDevice(index);
		cudaError_t err = cudaGetLastError();

		// Ignore errors caused by calling cudaSetDevice multiple times
		if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
			return false;
	}

	return true;
}

bool AstraSIRT3d::init()
{
	if (pData->initialized)
		return false;

	if (pData->dims.iVolX == 0 || pData->dims.iProjAngles == 0)
		return false;

	bool ok;

	if (pData->projType == PROJ_PARALLEL) {
		ok = pData->sirt.setPar3DGeometry(pData->dims, pData->parprojs);
	} else {
		ok = pData->sirt.setConeGeometry(pData->dims, pData->projs);
	}

	if (!ok)
		return false;

	ok = pData->sirt.init();
	if (!ok)
		return false;

	pData->D_volumeData = allocateVolumeData(pData->dims);
	ok = pData->D_volumeData.ptr;
	if (!ok)
		return false;

	pData->D_projData = allocateProjectionData(pData->dims);
	ok = pData->D_projData.ptr;
	if (!ok) {
		cudaFree(pData->D_volumeData.ptr);
		pData->D_volumeData.ptr = 0;
		return false;
	}

	if (pData->useVolumeMask) {
		pData->D_maskData = allocateVolumeData(pData->dims);
		ok = pData->D_maskData.ptr;
		if (!ok) {
			cudaFree(pData->D_volumeData.ptr);
			cudaFree(pData->D_projData.ptr);
			pData->D_volumeData.ptr = 0;
			pData->D_projData.ptr = 0;
			return false;
		}
	}

	if (pData->useSinogramMask) {
		pData->D_smaskData = allocateProjectionData(pData->dims);
		ok = pData->D_smaskData.ptr;
		if (!ok) {
			cudaFree(pData->D_volumeData.ptr);
			cudaFree(pData->D_projData.ptr);
			cudaFree(pData->D_maskData.ptr);
			pData->D_volumeData.ptr = 0;
			pData->D_projData.ptr = 0;
			pData->D_maskData.ptr = 0;
			return false;
		}
	}

	pData->initialized = true;

	return true;
}

bool AstraSIRT3d::setMinConstraint(float fMin)
{
	if (!pData->initialized)
		return false;
	return pData->sirt.setMinConstraint(fMin);
}

bool AstraSIRT3d::setMaxConstraint(float fMax)
{
	if (!pData->initialized)
		return false;
	return pData->sirt.setMaxConstraint(fMax);
}

bool AstraSIRT3d::setSinogram(const float* pfSinogram,
                              unsigned int iSinogramPitch)
{
	if (!pData->initialized)
		return false;
	if (!pfSinogram)
		return false;

	bool ok = copyProjectionsToDevice(pfSinogram, pData->D_projData, pData->dims, iSinogramPitch);

	if (!ok)
		return false;

	ok = pData->sirt.setBuffers(pData->D_volumeData, pData->D_projData);
	if (!ok)
		return false;

	pData->setStartReconstruction = false;

	return true;
}

bool AstraSIRT3d::setVolumeMask(const float* pfMask, unsigned int iMaskPitch)
{
	if (!pData->initialized)
		return false;
	if (!pData->useVolumeMask)
		return false;
	if (!pfMask)
		return false;

	bool ok = copyVolumeToDevice(pfMask, pData->D_maskData,
	                             pData->dims, iMaskPitch);
	if (!ok)
		return false;

	ok = pData->sirt.setVolumeMask(pData->D_maskData);
	if (!ok)
		return false;

	return true;
}

bool AstraSIRT3d::setSinogramMask(const float* pfMask, unsigned int iMaskPitch)
{
	if (!pData->initialized)
		return false;
	if (!pData->useSinogramMask)
		return false;
	if (!pfMask)
		return false;

	bool ok = copyProjectionsToDevice(pfMask, pData->D_smaskData, pData->dims, iMaskPitch);

	if (!ok)
		return false;

	ok = pData->sirt.setSinogramMask(pData->D_smaskData);
	if (!ok)
		return false;

	return true;
}

bool AstraSIRT3d::setStartReconstruction(const float* pfReconstruction,
                                         unsigned int iReconstructionPitch)
{
	if (!pData->initialized)
		return false;
	if (!pfReconstruction)
		return false;

	bool ok = copyVolumeToDevice(pfReconstruction, pData->D_volumeData,
	                             pData->dims, iReconstructionPitch);
	if (!ok)
		return false;

	pData->setStartReconstruction = true;

	return true;
}

bool AstraSIRT3d::iterate(unsigned int iIterations)
{
	if (!pData->initialized)
		return false;

	if (!pData->setStartReconstruction)
		zeroVolumeData(pData->D_volumeData, pData->dims);

	bool ok = pData->sirt.iterate(iIterations);
	if (!ok)
		return false;

	return true;
}

bool AstraSIRT3d::getReconstruction(float* pfReconstruction,
                                    unsigned int iReconstructionPitch) const
{
	if (!pData->initialized)
		return false;

	bool ok = copyVolumeFromDevice(pfReconstruction, pData->D_volumeData,
	                               pData->dims, iReconstructionPitch);
	if (!ok)
		return false;

	return true;
}

void AstraSIRT3d::signalAbort()
{
	if (!pData->initialized)
		return;

	pData->sirt.signalAbort();
}

float AstraSIRT3d::computeDiffNorm()
{
	if (!pData->initialized)
		return 0.0f; // FIXME: Error?

	return pData->sirt.computeDiffNorm();
}




class AstraCGLS3d_internal {
public:
	SDimensions3D dims;
	CUDAProjectionType3d projType;

	float* angles;
	float fOriginSourceDistance;
	float fOriginDetectorDistance;
	float fSourceZ;
	float fDetSize;

	SConeProjection* projs;
	SPar3DProjection* parprojs;

	float fPixelSize;

	bool initialized;
	bool setStartReconstruction;

	bool useVolumeMask;
	bool useSinogramMask;

	// Input/output
	cudaPitchedPtr D_projData;
	cudaPitchedPtr D_volumeData;
	cudaPitchedPtr D_maskData;
	cudaPitchedPtr D_smaskData;

	CGLS cgls;
};

AstraCGLS3d::AstraCGLS3d()
{
	pData = new AstraCGLS3d_internal();

	pData->angles = 0;
	pData->D_projData.ptr = 0;
	pData->D_volumeData.ptr = 0;
	pData->D_maskData.ptr = 0;
	pData->D_smaskData.ptr = 0;

	pData->dims.iVolX = 0;
	pData->dims.iVolY = 0;
	pData->dims.iVolZ = 0;
	pData->dims.iProjAngles = 0;
	pData->dims.iProjU = 0;
	pData->dims.iProjV = 0;
	pData->dims.iRaysPerDetDim = 1;
	pData->dims.iRaysPerVoxelDim = 1;

	pData->projs = 0;

	pData->initialized = false;
	pData->setStartReconstruction = false;

	pData->useVolumeMask = false;
	pData->useSinogramMask = false;
}

AstraCGLS3d::~AstraCGLS3d()
{
	delete[] pData->angles;
	pData->angles = 0;

	delete[] pData->projs;
	pData->projs = 0;

	cudaFree(pData->D_projData.ptr);
	pData->D_projData.ptr = 0;

	cudaFree(pData->D_volumeData.ptr);
	pData->D_volumeData.ptr = 0;

	cudaFree(pData->D_maskData.ptr);
	pData->D_maskData.ptr = 0;

	cudaFree(pData->D_smaskData.ptr);
	pData->D_smaskData.ptr = 0;

	delete pData;
	pData = 0;
}

bool AstraCGLS3d::setReconstructionGeometry(unsigned int iVolX,
                                            unsigned int iVolY,
                                            unsigned int iVolZ/*,
                                            float fPixelSize = 1.0f*/)
{
	if (pData->initialized)
		return false;

	pData->dims.iVolX = iVolX;
	pData->dims.iVolY = iVolY;
	pData->dims.iVolZ = iVolZ;

	return (iVolX > 0 && iVolY > 0 && iVolZ > 0);
}


bool AstraCGLS3d::setPar3DGeometry(unsigned int iProjAngles,
                                   unsigned int iProjU,
                                   unsigned int iProjV,
                                   const SPar3DProjection* projs)
{
	if (pData->initialized)
		return false;

	pData->dims.iProjAngles = iProjAngles;
	pData->dims.iProjU = iProjU;
	pData->dims.iProjV = iProjV;

	if (iProjAngles == 0 || iProjU == 0 || iProjV == 0 || projs == 0)
		return false;

	pData->parprojs = new SPar3DProjection[iProjAngles];
	memcpy(pData->parprojs, projs, iProjAngles * sizeof(projs[0]));

	pData->projType = PROJ_PARALLEL;

	return true;
}

bool AstraCGLS3d::setPar3DGeometry(unsigned int iProjAngles,
                                   unsigned int iProjU,
                                   unsigned int iProjV,
                                   float fDetUSize,
                                   float fDetVSize,
                                   const float *pfAngles)
{
	if (pData->initialized)
		return false;

	if (iProjAngles == 0 || iProjU == 0 || iProjV == 0 || pfAngles == 0)
		return false;

	SPar3DProjection* p = genPar3DProjections(iProjAngles,
                                              iProjU, iProjV,
                                              fDetUSize, fDetVSize,
                                              pfAngles);
	pData->dims.iProjAngles = iProjAngles;
	pData->dims.iProjU = iProjU;
	pData->dims.iProjV = iProjV;

	pData->parprojs = p;
	pData->projType = PROJ_PARALLEL;

	return true;
}



bool AstraCGLS3d::setConeGeometry(unsigned int iProjAngles,
                                  unsigned int iProjU,
                                  unsigned int iProjV,
                                  const SConeProjection* projs)
{
	if (pData->initialized)
		return false;

	pData->dims.iProjAngles = iProjAngles;
	pData->dims.iProjU = iProjU;
	pData->dims.iProjV = iProjV;

	if (iProjAngles == 0 || iProjU == 0 || iProjV == 0 || projs == 0)
		return false;

	pData->projs = new SConeProjection[iProjAngles];
	memcpy(pData->projs, projs, iProjAngles * sizeof(projs[0]));

	pData->projType = PROJ_CONE;

	return true;
}

bool AstraCGLS3d::setConeGeometry(unsigned int iProjAngles,
                                  unsigned int iProjU,
                                  unsigned int iProjV,
                                  float fOriginSourceDistance,
                                  float fOriginDetectorDistance,
                                  float fDetUSize,
                                  float fDetVSize,
                                  const float *pfAngles)
{
	if (pData->initialized)
		return false;

	if (iProjAngles == 0 || iProjU == 0 || iProjV == 0 || pfAngles == 0)
		return false;

	SConeProjection* p = genConeProjections(iProjAngles,
                                            iProjU, iProjV,
                                            fOriginSourceDistance,
                                            fOriginDetectorDistance,
                                            fDetUSize, fDetVSize,
                                            pfAngles);

	pData->dims.iProjAngles = iProjAngles;
	pData->dims.iProjU = iProjU;
	pData->dims.iProjV = iProjV;

	pData->projs = p;
	pData->projType = PROJ_CONE;

	return true;
}

bool AstraCGLS3d::enableSuperSampling(unsigned int iVoxelSuperSampling,
                                      unsigned int iDetectorSuperSampling)
{
	if (pData->initialized)
		return false;

	if (iVoxelSuperSampling == 0 || iDetectorSuperSampling == 0)
		return false;

	pData->dims.iRaysPerVoxelDim = iVoxelSuperSampling;
	pData->dims.iRaysPerDetDim = iDetectorSuperSampling;

	return true;
}

bool AstraCGLS3d::enableVolumeMask()
{
	if (pData->initialized)
		return false;

	bool ok = pData->cgls.enableVolumeMask();
	pData->useVolumeMask = ok;

	return ok;
}

#if 0
bool AstraCGLS3d::enableSinogramMask()
{
	if (pData->initialized)
		return false;

	bool ok = pData->cgls.enableSinogramMask();
	pData->useSinogramMask = ok;

	return ok;
}
#endif
	
bool AstraCGLS3d::setGPUIndex(int index)
{
	if (index != -1) {
		cudaSetDevice(index);
		cudaError_t err = cudaGetLastError();

		// Ignore errors caused by calling cudaSetDevice multiple times
		if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
			return false;
	}

	return true;
}

bool AstraCGLS3d::init()
{
	if (pData->initialized)
		return false;

	if (pData->dims.iVolX == 0 || pData->dims.iProjAngles == 0)
		return false;

	bool ok;

	if (pData->projType == PROJ_PARALLEL) {
		ok = pData->cgls.setPar3DGeometry(pData->dims, pData->parprojs);
	} else {
		ok = pData->cgls.setConeGeometry(pData->dims, pData->projs);
	}

	if (!ok)
		return false;

	ok = pData->cgls.init();
	if (!ok)
		return false;

	pData->D_volumeData = allocateVolumeData(pData->dims);
	ok = pData->D_volumeData.ptr;
	if (!ok)
		return false;

	pData->D_projData = allocateProjectionData(pData->dims);
	ok = pData->D_projData.ptr;
	if (!ok) {
		cudaFree(pData->D_volumeData.ptr);
		pData->D_volumeData.ptr = 0;
		return false;
	}

	if (pData->useVolumeMask) {
		pData->D_maskData = allocateVolumeData(pData->dims);
		ok = pData->D_maskData.ptr;
		if (!ok) {
			cudaFree(pData->D_volumeData.ptr);
			cudaFree(pData->D_projData.ptr);
			pData->D_volumeData.ptr = 0;
			pData->D_projData.ptr = 0;
			return false;
		}
	}

	if (pData->useSinogramMask) {
		pData->D_smaskData = allocateProjectionData(pData->dims);
		ok = pData->D_smaskData.ptr;
		if (!ok) {
			cudaFree(pData->D_volumeData.ptr);
			cudaFree(pData->D_projData.ptr);
			cudaFree(pData->D_maskData.ptr);
			pData->D_volumeData.ptr = 0;
			pData->D_projData.ptr = 0;
			pData->D_maskData.ptr = 0;
			return false;
		}
	}

	pData->initialized = true;

	return true;
}

#if 0
bool AstraCGLS3d::setMinConstraint(float fMin)
{
	if (!pData->initialized)
		return false;
	return pData->cgls.setMinConstraint(fMin);
}

bool AstraCGLS3d::setMaxConstraint(float fMax)
{
	if (!pData->initialized)
		return false;
	return pData->cgls.setMaxConstraint(fMax);
}
#endif

bool AstraCGLS3d::setSinogram(const float* pfSinogram,
                              unsigned int iSinogramPitch)
{
	if (!pData->initialized)
		return false;
	if (!pfSinogram)
		return false;

	bool ok = copyProjectionsToDevice(pfSinogram, pData->D_projData, pData->dims, iSinogramPitch);

	if (!ok)
		return false;

	ok = pData->cgls.setBuffers(pData->D_volumeData, pData->D_projData);
	if (!ok)
		return false;

	pData->setStartReconstruction = false;

	return true;
}

bool AstraCGLS3d::setVolumeMask(const float* pfMask, unsigned int iMaskPitch)
{
	if (!pData->initialized)
		return false;
	if (!pData->useVolumeMask)
		return false;
	if (!pfMask)
		return false;

	bool ok = copyVolumeToDevice(pfMask, pData->D_maskData,
	                             pData->dims, iMaskPitch);
	if (!ok)
		return false;

	ok = pData->cgls.setVolumeMask(pData->D_maskData);
	if (!ok)
		return false;

	return true;
}

#if 0
bool AstraCGLS3d::setSinogramMask(const float* pfMask, unsigned int iMaskPitch)
{
	if (!pData->initialized)
		return false;
	if (!pData->useSinogramMask)
		return false;
	if (!pfMask)
		return false;

	bool ok = copyProjectionsToDevice(pfMask, pData->D_smaskData, pData->dims, iMaskPitch);

	if (!ok)
		return false;

	ok = pData->cgls.setSinogramMask(pData->D_smaskData);
	if (!ok)
		return false;

	return true;
}
#endif

bool AstraCGLS3d::setStartReconstruction(const float* pfReconstruction,
                                         unsigned int iReconstructionPitch)
{
	if (!pData->initialized)
		return false;
	if (!pfReconstruction)
		return false;

	bool ok = copyVolumeToDevice(pfReconstruction, pData->D_volumeData,
	                             pData->dims, iReconstructionPitch);
	if (!ok)
		return false;

	pData->setStartReconstruction = true;

	return true;
}

bool AstraCGLS3d::iterate(unsigned int iIterations)
{
	if (!pData->initialized)
		return false;

	if (!pData->setStartReconstruction)
		zeroVolumeData(pData->D_volumeData, pData->dims);

	bool ok = pData->cgls.iterate(iIterations);
	if (!ok)
		return false;

	return true;
}

bool AstraCGLS3d::getReconstruction(float* pfReconstruction,
                                    unsigned int iReconstructionPitch) const
{
	if (!pData->initialized)
		return false;

	bool ok = copyVolumeFromDevice(pfReconstruction, pData->D_volumeData,
	                               pData->dims, iReconstructionPitch);
	if (!ok)
		return false;

	return true;
}

void AstraCGLS3d::signalAbort()
{
	if (!pData->initialized)
		return;

	pData->cgls.signalAbort();
}

float AstraCGLS3d::computeDiffNorm()
{
	if (!pData->initialized)
		return 0.0f; // FIXME: Error?

	return pData->cgls.computeDiffNorm();
}



bool astraCudaConeFP(const float* pfVolume, float* pfProjections,
                     unsigned int iVolX,
                     unsigned int iVolY,
                     unsigned int iVolZ,
                     unsigned int iProjAngles,
                     unsigned int iProjU,
                     unsigned int iProjV,
                     float fOriginSourceDistance,
                     float fOriginDetectorDistance,
                     float fDetUSize,
                     float fDetVSize,
                     const float *pfAngles,
                     int iGPUIndex, int iDetectorSuperSampling)
{
	if (iVolX == 0 || iVolY == 0 || iVolZ == 0)
		return false;
	if (iProjAngles == 0 || iProjU == 0 || iProjV == 0 || pfAngles == 0)
		return false;

	SConeProjection* p = genConeProjections(iProjAngles,
                                            iProjU, iProjV,
                                            fOriginSourceDistance,
                                            fOriginDetectorDistance,
                                            fDetUSize, fDetVSize,
                                            pfAngles);

	bool ok;
	ok = astraCudaConeFP(pfVolume, pfProjections, iVolX, iVolY, iVolZ,
	                     iProjAngles, iProjU, iProjV, p, iGPUIndex, iDetectorSuperSampling);

	delete[] p;

	return ok;
}

bool astraCudaConeFP(const float* pfVolume, float* pfProjections,
                     unsigned int iVolX,
                     unsigned int iVolY,
                     unsigned int iVolZ,
                     unsigned int iProjAngles,
                     unsigned int iProjU,
                     unsigned int iProjV,
                     const SConeProjection *pfAngles,
                     int iGPUIndex, int iDetectorSuperSampling)
{
	SDimensions3D dims;

	dims.iVolX = iVolX;
	dims.iVolY = iVolY;
	dims.iVolZ = iVolZ;
	if (iVolX == 0 || iVolY == 0 || iVolZ == 0)
		return false;

	dims.iProjAngles = iProjAngles;
	dims.iProjU = iProjU;
	dims.iProjV = iProjV;

	if (iProjAngles == 0 || iProjU == 0 || iProjV == 0 || pfAngles == 0)
		return false;

	dims.iRaysPerDetDim = iDetectorSuperSampling;

	if (iDetectorSuperSampling == 0)
		return false;

	if (iGPUIndex != -1) {
		cudaSetDevice(iGPUIndex);
		cudaError_t err = cudaGetLastError();

		// Ignore errors caused by calling cudaSetDevice multiple times
		if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
			return false;
	}

	cudaPitchedPtr D_volumeData = allocateVolumeData(dims);
	bool ok = D_volumeData.ptr;
	if (!ok)
		return false;

	cudaPitchedPtr D_projData = allocateProjectionData(dims);
	ok = D_projData.ptr;
	if (!ok) {
		cudaFree(D_volumeData.ptr);
		return false;
	}

	ok &= copyVolumeToDevice(pfVolume, D_volumeData, dims, dims.iVolX);

	ok &= zeroProjectionData(D_projData, dims);

	if (!ok) {
		cudaFree(D_volumeData.ptr);
		cudaFree(D_projData.ptr);
		return false;
	}

	ok &= ConeFP(D_volumeData, D_projData, dims, pfAngles, 1.0f);

	ok &= copyProjectionsFromDevice(pfProjections, D_projData,
	                                dims, dims.iProjU);


	cudaFree(D_volumeData.ptr);
	cudaFree(D_projData.ptr);

	return ok;

}

bool astraCudaPar3DFP(const float* pfVolume, float* pfProjections,
                      unsigned int iVolX,
                      unsigned int iVolY,
                      unsigned int iVolZ,
                      unsigned int iProjAngles,
                      unsigned int iProjU,
                      unsigned int iProjV,
                      float fDetUSize,
                      float fDetVSize,
                      const float *pfAngles,
                      int iGPUIndex, int iDetectorSuperSampling,
                      Cuda3DProjectionKernel projKernel)
{
	if (iVolX == 0 || iVolY == 0 || iVolZ == 0)
		return false;
	if (iProjAngles == 0 || iProjU == 0 || iProjV == 0 || pfAngles == 0)
		return false;

	SPar3DProjection* p = genPar3DProjections(iProjAngles,
                                             iProjU, iProjV,
                                             fDetUSize, fDetVSize,
                                             pfAngles);

	bool ok;
	ok = astraCudaPar3DFP(pfVolume, pfProjections, iVolX, iVolY, iVolZ,
	                      iProjAngles, iProjU, iProjV, p, iGPUIndex, iDetectorSuperSampling,
	                      projKernel);

	delete[] p;

	return ok;
}


bool astraCudaPar3DFP(const float* pfVolume, float* pfProjections,
                      unsigned int iVolX,
                      unsigned int iVolY,
                      unsigned int iVolZ,
                      unsigned int iProjAngles,
                      unsigned int iProjU,
                      unsigned int iProjV,
                      const SPar3DProjection *pfAngles,
                      int iGPUIndex, int iDetectorSuperSampling,
                      Cuda3DProjectionKernel projKernel)
{
	SDimensions3D dims;

	dims.iVolX = iVolX;
	dims.iVolY = iVolY;
	dims.iVolZ = iVolZ;
	if (iVolX == 0 || iVolY == 0 || iVolZ == 0)
		return false;

	dims.iProjAngles = iProjAngles;
	dims.iProjU = iProjU;
	dims.iProjV = iProjV;

	if (iProjAngles == 0 || iProjU == 0 || iProjV == 0 || pfAngles == 0)
		return false;

	dims.iRaysPerDetDim = iDetectorSuperSampling;

	if (iDetectorSuperSampling == 0)
		return false;

	if (iGPUIndex != -1) {
		cudaSetDevice(iGPUIndex);
		cudaError_t err = cudaGetLastError();

		// Ignore errors caused by calling cudaSetDevice multiple times
		if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
			return false;
	}


	cudaPitchedPtr D_volumeData = allocateVolumeData(dims);
	bool ok = D_volumeData.ptr;
	if (!ok)
		return false;

	cudaPitchedPtr D_projData = allocateProjectionData(dims);
	ok = D_projData.ptr;
	if (!ok) {
		cudaFree(D_volumeData.ptr);
		return false;
	}

	ok &= copyVolumeToDevice(pfVolume, D_volumeData, dims, dims.iVolX);

	ok &= zeroProjectionData(D_projData, dims);

	if (!ok) {
		cudaFree(D_volumeData.ptr);
		cudaFree(D_projData.ptr);
		return false;
	}

	switch (projKernel) {
	case ker3d_default:
		ok &= Par3DFP(D_volumeData, D_projData, dims, pfAngles, 1.0f);
		break;
	case ker3d_sum_square_weights:
		ok &= Par3DFP_SumSqW(D_volumeData, D_projData, dims, pfAngles, 1.0f);
		break;
	default:
		assert(false);
	}

	ok &= copyProjectionsFromDevice(pfProjections, D_projData,
	                                dims, dims.iProjU);


	cudaFree(D_volumeData.ptr);
	cudaFree(D_projData.ptr);

	return ok;

}

bool astraCudaConeBP(float* pfVolume, const float* pfProjections,
                     unsigned int iVolX,
                     unsigned int iVolY,
                     unsigned int iVolZ,
                     unsigned int iProjAngles,
                     unsigned int iProjU,
                     unsigned int iProjV,
                     float fOriginSourceDistance,
                     float fOriginDetectorDistance,
                     float fDetUSize,
                     float fDetVSize,
                     const float *pfAngles,
                     int iGPUIndex, int iVoxelSuperSampling)
{
	if (iVolX == 0 || iVolY == 0 || iVolZ == 0)
		return false;
	if (iProjAngles == 0 || iProjU == 0 || iProjV == 0 || pfAngles == 0)
		return false;

	SConeProjection* p = genConeProjections(iProjAngles,
                                            iProjU, iProjV,
                                            fOriginSourceDistance,
                                            fOriginDetectorDistance,
                                            fDetUSize, fDetVSize,
                                            pfAngles);

	bool ok;
	ok = astraCudaConeBP(pfVolume, pfProjections, iVolX, iVolY, iVolZ,
	                     iProjAngles, iProjU, iProjV, p, iGPUIndex, iVoxelSuperSampling);

	delete[] p;

	return ok;
}

bool astraCudaConeBP(float* pfVolume, const float* pfProjections,
                     unsigned int iVolX,
                     unsigned int iVolY,
                     unsigned int iVolZ,
                     unsigned int iProjAngles,
                     unsigned int iProjU,
                     unsigned int iProjV,
                     const SConeProjection *pfAngles,
                     int iGPUIndex, int iVoxelSuperSampling)
{
	SDimensions3D dims;

	dims.iVolX = iVolX;
	dims.iVolY = iVolY;
	dims.iVolZ = iVolZ;
	if (iVolX == 0 || iVolY == 0 || iVolZ == 0)
		return false;

	dims.iProjAngles = iProjAngles;
	dims.iProjU = iProjU;
	dims.iProjV = iProjV;

	if (iProjAngles == 0 || iProjU == 0 || iProjV == 0 || pfAngles == 0)
		return false;

	dims.iRaysPerVoxelDim = iVoxelSuperSampling;

	if (iProjAngles == 0 || iProjU == 0 || iProjV == 0 || pfAngles == 0)
		return false;

	if (iGPUIndex != -1) {
		cudaSetDevice(iGPUIndex);
		cudaError_t err = cudaGetLastError();

		// Ignore errors caused by calling cudaSetDevice multiple times
		if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
			return false;
	}

	cudaPitchedPtr D_volumeData = allocateVolumeData(dims);
	bool ok = D_volumeData.ptr;
	if (!ok)
		return false;

	cudaPitchedPtr D_projData = allocateProjectionData(dims);
	ok = D_projData.ptr;
	if (!ok) {
		cudaFree(D_volumeData.ptr);
		return false;
	}

	ok &= copyProjectionsToDevice(pfProjections, D_projData,
	                              dims, dims.iProjU);

	ok &= zeroVolumeData(D_volumeData, dims);

	if (!ok) {
		cudaFree(D_volumeData.ptr);
		cudaFree(D_projData.ptr);
		return false;
	}

	ok &= ConeBP(D_volumeData, D_projData, dims, pfAngles);

	ok &= copyVolumeFromDevice(pfVolume, D_volumeData, dims, dims.iVolX);


	cudaFree(D_volumeData.ptr);
	cudaFree(D_projData.ptr);

	return ok;

}

bool astraCudaPar3DBP(float* pfVolume, const float* pfProjections,
                      unsigned int iVolX,
                      unsigned int iVolY,
                      unsigned int iVolZ,
                      unsigned int iProjAngles,
                      unsigned int iProjU,
                      unsigned int iProjV,
                      float fDetUSize,
                      float fDetVSize,
                      const float *pfAngles,
                      int iGPUIndex, int iVoxelSuperSampling)
{
	if (iVolX == 0 || iVolY == 0 || iVolZ == 0)
		return false;
	if (iProjAngles == 0 || iProjU == 0 || iProjV == 0 || pfAngles == 0)
		return false;

	SPar3DProjection* p = genPar3DProjections(iProjAngles,
                                             iProjU, iProjV,
                                             fDetUSize, fDetVSize,
                                             pfAngles);

	bool ok;
	ok = astraCudaPar3DBP(pfVolume, pfProjections, iVolX, iVolY, iVolZ,
	                      iProjAngles, iProjU, iProjV, p, iGPUIndex, iVoxelSuperSampling);

	delete[] p;

	return ok;
}

// This computes the column weights, divides by them, and adds the
// result to the current volume. This is both more expensive and more
// GPU memory intensive than the regular BP, but allows saving system RAM.
bool astraCudaPar3DBP_SIRTWeighted(float* pfVolume, const float* pfProjections,
                      unsigned int iVolX,
                      unsigned int iVolY,
                      unsigned int iVolZ,
                      unsigned int iProjAngles,
                      unsigned int iProjU,
                      unsigned int iProjV,
                      float fDetUSize,
                      float fDetVSize,
                      const float *pfAngles,
                      int iGPUIndex, int iVoxelSuperSampling)
{
	if (iVolX == 0 || iVolY == 0 || iVolZ == 0)
		return false;
	if (iProjAngles == 0 || iProjU == 0 || iProjV == 0 || pfAngles == 0)
		return false;

	SPar3DProjection* p = genPar3DProjections(iProjAngles,
                                             iProjU, iProjV,
                                             fDetUSize, fDetVSize,
                                             pfAngles);

	bool ok;
	ok = astraCudaPar3DBP_SIRTWeighted(pfVolume, pfProjections, iVolX, iVolY, iVolZ,
	                      iProjAngles, iProjU, iProjV, p, iGPUIndex, iVoxelSuperSampling);

	delete[] p;

	return ok;
}


bool astraCudaPar3DBP(float* pfVolume, const float* pfProjections,
                      unsigned int iVolX,
                      unsigned int iVolY,
                      unsigned int iVolZ,
                      unsigned int iProjAngles,
                      unsigned int iProjU,
                      unsigned int iProjV,
                      const SPar3DProjection *pfAngles,
                      int iGPUIndex, int iVoxelSuperSampling)
{
	SDimensions3D dims;

	dims.iVolX = iVolX;
	dims.iVolY = iVolY;
	dims.iVolZ = iVolZ;
	if (iVolX == 0 || iVolY == 0 || iVolZ == 0)
		return false;

	dims.iProjAngles = iProjAngles;
	dims.iProjU = iProjU;
	dims.iProjV = iProjV;

	if (iProjAngles == 0 || iProjU == 0 || iProjV == 0 || pfAngles == 0)
		return false;

	dims.iRaysPerVoxelDim = iVoxelSuperSampling;

	if (iGPUIndex != -1) {
		cudaSetDevice(iGPUIndex);
		cudaError_t err = cudaGetLastError();

		// Ignore errors caused by calling cudaSetDevice multiple times
		if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
			return false;
	}


	cudaPitchedPtr D_volumeData = allocateVolumeData(dims);
	bool ok = D_volumeData.ptr;
	if (!ok)
		return false;

	cudaPitchedPtr D_projData = allocateProjectionData(dims);
	ok = D_projData.ptr;
	if (!ok) {
		cudaFree(D_volumeData.ptr);
		return false;
	}

	ok &= copyProjectionsToDevice(pfProjections, D_projData,
	                              dims, dims.iProjU);

	ok &= zeroVolumeData(D_volumeData, dims);

	if (!ok) {
		cudaFree(D_volumeData.ptr);
		cudaFree(D_projData.ptr);
		return false;
	}

	ok &= Par3DBP(D_volumeData, D_projData, dims, pfAngles);

	ok &= copyVolumeFromDevice(pfVolume, D_volumeData, dims, dims.iVolX);


	cudaFree(D_volumeData.ptr);
	cudaFree(D_projData.ptr);

	return ok;

}


// This computes the column weights, divides by them, and adds the
// result to the current volume. This is both more expensive and more
// GPU memory intensive than the regular BP, but allows saving system RAM.
bool astraCudaPar3DBP_SIRTWeighted(float* pfVolume,
                      const float* pfProjections,
                      unsigned int iVolX,
                      unsigned int iVolY,
                      unsigned int iVolZ,
                      unsigned int iProjAngles,
                      unsigned int iProjU,
                      unsigned int iProjV,
                      const SPar3DProjection *pfAngles,
                      int iGPUIndex, int iVoxelSuperSampling)
{
	SDimensions3D dims;

	dims.iVolX = iVolX;
	dims.iVolY = iVolY;
	dims.iVolZ = iVolZ;
	if (iVolX == 0 || iVolY == 0 || iVolZ == 0)
		return false;

	dims.iProjAngles = iProjAngles;
	dims.iProjU = iProjU;
	dims.iProjV = iProjV;

	if (iProjAngles == 0 || iProjU == 0 || iProjV == 0 || pfAngles == 0)
		return false;

	dims.iRaysPerVoxelDim = iVoxelSuperSampling;

	if (iGPUIndex != -1) {
		cudaSetDevice(iGPUIndex);
		cudaError_t err = cudaGetLastError();

		// Ignore errors caused by calling cudaSetDevice multiple times
		if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
			return false;
	}


	cudaPitchedPtr D_pixelWeight = allocateVolumeData(dims);
	bool ok = D_pixelWeight.ptr;
	if (!ok)
		return false;

	cudaPitchedPtr D_volumeData = allocateVolumeData(dims);
	ok = D_volumeData.ptr;
	if (!ok) {
		cudaFree(D_pixelWeight.ptr);
		return false;
	}

	cudaPitchedPtr D_projData = allocateProjectionData(dims);
	ok = D_projData.ptr;
	if (!ok) {
		cudaFree(D_pixelWeight.ptr);
		cudaFree(D_volumeData.ptr);
		return false;
	}

	// Compute weights
	ok &= zeroVolumeData(D_pixelWeight, dims);
	processSino3D<opSet>(D_projData, 1.0f, dims);
	ok &= Par3DBP(D_pixelWeight, D_projData, dims, pfAngles);
	processVol3D<opInvert>(D_pixelWeight, dims);
	if (!ok) {
		cudaFree(D_pixelWeight.ptr);
		cudaFree(D_volumeData.ptr);
		cudaFree(D_projData.ptr);
		return false;
	}

	ok &= copyProjectionsToDevice(pfProjections, D_projData,
	                              dims, dims.iProjU);
	ok &= zeroVolumeData(D_volumeData, dims);
	// Do BP into D_volumeData
	ok &= Par3DBP(D_volumeData, D_projData, dims, pfAngles);
	// Multiply with weights
	processVol3D<opMul>(D_volumeData, D_pixelWeight, dims);

	// Upload previous iterate to D_pixelWeight...
	ok &= copyVolumeToDevice(pfVolume, D_pixelWeight, dims, dims.iVolX);
	if (!ok) {
		cudaFree(D_pixelWeight.ptr);
		cudaFree(D_volumeData.ptr);
		cudaFree(D_projData.ptr);
		return false;
	}
	// ...and add it to the weighted BP
	processVol3D<opAdd>(D_volumeData, D_pixelWeight, dims);

	// Then copy the result back
	ok &= copyVolumeFromDevice(pfVolume, D_volumeData, dims, dims.iVolX);


	cudaFree(D_pixelWeight.ptr);
	cudaFree(D_volumeData.ptr);
	cudaFree(D_projData.ptr);

	return ok;

}



bool astraCudaFDK(float* pfVolume, const float* pfProjections,
                  unsigned int iVolX,
                  unsigned int iVolY,
                  unsigned int iVolZ,
                  unsigned int iProjAngles,
                  unsigned int iProjU,
                  unsigned int iProjV,
                  float fOriginSourceDistance,
                  float fOriginDetectorDistance,
                  float fDetUSize,
                  float fDetVSize,
                  const float *pfAngles,
                  bool bShortScan,
                  int iGPUIndex, int iVoxelSuperSampling)
{
	SDimensions3D dims;

	dims.iVolX = iVolX;
	dims.iVolY = iVolY;
	dims.iVolZ = iVolZ;
	if (iVolX == 0 || iVolY == 0 || iVolZ == 0)
		return false;

	dims.iProjAngles = iProjAngles;
	dims.iProjU = iProjU;
	dims.iProjV = iProjV;

	if (iProjAngles == 0 || iProjU == 0 || iProjV == 0 || pfAngles == 0)
		return false;

	dims.iRaysPerVoxelDim = iVoxelSuperSampling;

	if (iVoxelSuperSampling == 0)
		return false;

	if (iGPUIndex != -1) {
		cudaSetDevice(iGPUIndex);
		cudaError_t err = cudaGetLastError();

		// Ignore errors caused by calling cudaSetDevice multiple times
		if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
			return false;
	}


	cudaPitchedPtr D_volumeData = allocateVolumeData(dims);
	bool ok = D_volumeData.ptr;
	if (!ok)
		return false;

	cudaPitchedPtr D_projData = allocateProjectionData(dims);
	ok = D_projData.ptr;
	if (!ok) {
		cudaFree(D_volumeData.ptr);
		return false;
	}

	ok &= copyProjectionsToDevice(pfProjections, D_projData, dims, dims.iProjU);

	ok &= zeroVolumeData(D_volumeData, dims);

	if (!ok) {
		cudaFree(D_volumeData.ptr);
		cudaFree(D_projData.ptr);
		return false;
	}

	// TODO: Offer interface for SrcZ, DetZ
	ok &= FDK(D_volumeData, D_projData, fOriginSourceDistance,
	          fOriginDetectorDistance, 0, 0, fDetUSize, fDetVSize,
	          dims, pfAngles, bShortScan);

	ok &= copyVolumeFromDevice(pfVolume, D_volumeData, dims, dims.iVolX);


	cudaFree(D_volumeData.ptr);
	cudaFree(D_projData.ptr);

	return ok;

}




}
