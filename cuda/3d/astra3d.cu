/*
-----------------------------------------------------------------------
Copyright: 2010-2022, imec Vision Lab, University of Antwerp
           2014-2022, CWI, Amsterdam

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
#include "astra/cuda/3d/sirt3d.h"
#include "astra/cuda/3d/util3d.h"
#include "astra/cuda/3d/cone_fp.h"
#include "astra/cuda/3d/cone_bp.h"
#include "astra/cuda/3d/par3d_fp.h"
#include "astra/cuda/3d/par3d_bp.h"
#include "astra/cuda/3d/fdk.h"
#include "astra/cuda/3d/arith3d.h"
#include "astra/cuda/3d/astra3d.h"
#include "astra/cuda/3d/mem3d.h"

#include "astra/ParallelProjectionGeometry3D.h"
#include "astra/ParallelVecProjectionGeometry3D.h"
#include "astra/ConeProjectionGeometry3D.h"
#include "astra/ConeVecProjectionGeometry3D.h"
#include "astra/VolumeGeometry3D.h"
#include "astra/Data3D.h"
#include "astra/Logging.h"

#include <iostream>
#include <cstdio>
#include <cassert>

using namespace astraCUDA3d;

namespace astra {

enum CUDAProjectionType3d {
	PROJ_PARALLEL,
	PROJ_CONE
};






// adjust pProjs to normalize volume geometry
template<typename ProjectionT>
static bool convertAstraGeometry_internal(const CVolumeGeometry3D* pVolGeom,
                          unsigned int iProjectionAngleCount,
                          ProjectionT*& pProjs,
                          SProjectorParams3D& params)
{
	assert(pVolGeom);
	assert(pProjs);

#if 0
	// TODO: Relative instead of absolute
	const float EPS = 0.00001f;
	if (abs(pVolGeom->getPixelLengthX() - pVolGeom->getPixelLengthY()) > EPS)
		return false;
	if (abs(pVolGeom->getPixelLengthX() - pVolGeom->getPixelLengthZ()) > EPS)
		return false;
#endif

	// Translate
	float dx = -(pVolGeom->getWindowMinX() + pVolGeom->getWindowMaxX()) / 2;
	float dy = -(pVolGeom->getWindowMinY() + pVolGeom->getWindowMaxY()) / 2;
	float dz = -(pVolGeom->getWindowMinZ() + pVolGeom->getWindowMaxZ()) / 2;

	float fx = 1.0f / pVolGeom->getPixelLengthX();
	float fy = 1.0f / pVolGeom->getPixelLengthY();
	float fz = 1.0f / pVolGeom->getPixelLengthZ();

	for (int i = 0; i < iProjectionAngleCount; ++i) {
		// CHECKME: Order of scaling and translation
		pProjs[i].translate(dx, dy, dz);
		pProjs[i].scale(fx, fy, fz);
	}

	params.fVolScaleX = pVolGeom->getPixelLengthX();
	params.fVolScaleY = pVolGeom->getPixelLengthY();
	params.fVolScaleZ = pVolGeom->getPixelLengthZ();

	// CHECKME: Check factor
	//params.fOutputScale *= pVolGeom->getPixelLengthX();

	return true;
}


bool convertAstraGeometry_dims(const CVolumeGeometry3D* pVolGeom,
                               const CProjectionGeometry3D* pProjGeom,
                               SDimensions3D& dims)
{
	dims.iVolX = pVolGeom->getGridColCount();
	dims.iVolY = pVolGeom->getGridRowCount();
	dims.iVolZ = pVolGeom->getGridSliceCount();
	dims.iProjAngles = pProjGeom->getProjectionCount();
	dims.iProjU = pProjGeom->getDetectorColCount();
	dims.iProjV = pProjGeom->getDetectorRowCount();

	if (dims.iVolX <= 0 || dims.iVolX <= 0 || dims.iVolX <= 0)
		return false;
	if (dims.iProjAngles <= 0 || dims.iProjU <= 0 || dims.iProjV <= 0)
		return false;

	return true;
}


bool convertAstraGeometry(const CVolumeGeometry3D* pVolGeom,
                          const CParallelProjectionGeometry3D* pProjGeom,
                          SPar3DProjection*& pProjs, SProjectorParams3D& params)
{
	assert(pVolGeom);
	assert(pProjGeom);
	assert(pProjGeom->getProjectionAngles());

	int nth = pProjGeom->getProjectionCount();

	pProjs = genPar3DProjections(nth,
	                             pProjGeom->getDetectorColCount(),
	                             pProjGeom->getDetectorRowCount(),
	                             pProjGeom->getDetectorSpacingX(),
	                             pProjGeom->getDetectorSpacingY(),
	                             pProjGeom->getProjectionAngles());

	bool ok;

	ok = convertAstraGeometry_internal(pVolGeom, nth, pProjs, params);

	return ok;
}

bool convertAstraGeometry(const CVolumeGeometry3D* pVolGeom,
                          const CParallelVecProjectionGeometry3D* pProjGeom,
                          SPar3DProjection*& pProjs, SProjectorParams3D& params)
{
	assert(pVolGeom);
	assert(pProjGeom);
	assert(pProjGeom->getProjectionVectors());

	int nth = pProjGeom->getProjectionCount();

	pProjs = new SPar3DProjection[nth];
	for (int i = 0; i < nth; ++i)
		pProjs[i] = pProjGeom->getProjectionVectors()[i];

	bool ok;

	ok = convertAstraGeometry_internal(pVolGeom, nth, pProjs, params);

	return ok;
}

bool convertAstraGeometry(const CVolumeGeometry3D* pVolGeom,
                          const CConeProjectionGeometry3D* pProjGeom,
                          SConeProjection*& pProjs, SProjectorParams3D& params)
{
	assert(pVolGeom);
	assert(pProjGeom);
	assert(pProjGeom->getProjectionAngles());

	int nth = pProjGeom->getProjectionCount();

	pProjs = genConeProjections(nth,
	                            pProjGeom->getDetectorColCount(),
	                            pProjGeom->getDetectorRowCount(),
	                            pProjGeom->getOriginSourceDistance(),
	                            pProjGeom->getOriginDetectorDistance(),
	                            pProjGeom->getDetectorSpacingX(),
	                            pProjGeom->getDetectorSpacingY(),
	                            pProjGeom->getProjectionAngles());

	bool ok;

	ok = convertAstraGeometry_internal(pVolGeom, nth, pProjs, params);

	return ok;
}

bool convertAstraGeometry(const CVolumeGeometry3D* pVolGeom,
                          const CConeVecProjectionGeometry3D* pProjGeom,
                          SConeProjection*& pProjs, SProjectorParams3D& params)
{
	assert(pVolGeom);
	assert(pProjGeom);
	assert(pProjGeom->getProjectionVectors());

	int nth = pProjGeom->getProjectionCount();

	pProjs = new SConeProjection[nth];
	for (int i = 0; i < nth; ++i)
		pProjs[i] = pProjGeom->getProjectionVectors()[i];

	bool ok;

	ok = convertAstraGeometry_internal(pVolGeom, nth, pProjs, params);

	return ok;
}


bool convertAstraGeometry(const CVolumeGeometry3D* pVolGeom,
                          const CProjectionGeometry3D* pProjGeom,
                          SPar3DProjection*& pParProjs,
                          SConeProjection*& pConeProjs,
                          SProjectorParams3D& params)
{
	const CConeProjectionGeometry3D* conegeom = dynamic_cast<const CConeProjectionGeometry3D*>(pProjGeom);
	const CParallelProjectionGeometry3D* par3dgeom = dynamic_cast<const CParallelProjectionGeometry3D*>(pProjGeom);
	const CParallelVecProjectionGeometry3D* parvec3dgeom = dynamic_cast<const CParallelVecProjectionGeometry3D*>(pProjGeom);
	const CConeVecProjectionGeometry3D* conevec3dgeom = dynamic_cast<const CConeVecProjectionGeometry3D*>(pProjGeom);

	pConeProjs = 0;
	pParProjs = 0;

	bool ok;

	if (conegeom) {
		ok = convertAstraGeometry(pVolGeom, conegeom, pConeProjs, params);
	} else if (conevec3dgeom) {
		ok = convertAstraGeometry(pVolGeom, conevec3dgeom, pConeProjs, params);
	} else if (par3dgeom) {
		ok = convertAstraGeometry(pVolGeom, par3dgeom, pParProjs, params);
	} else if (parvec3dgeom) {
		ok = convertAstraGeometry(pVolGeom, parvec3dgeom, pParProjs, params);
	} else {
		ok = false;
	}

	return ok;
}




class AstraSIRT3d_internal {
public:
	SDimensions3D dims;
	SProjectorParams3D params;
	CUDAProjectionType3d projType;

	float* angles;
	float fOriginSourceDistance;
	float fOriginDetectorDistance;
	float fRelaxation;

	SConeProjection* projs;
	SPar3DProjection* parprojs;

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

	pData->projs = 0;
	pData->parprojs = 0;

	pData->fRelaxation = 1.0f;

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

	delete[] pData->parprojs;
	pData->parprojs = 0;

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

bool AstraSIRT3d::setGeometry(const CVolumeGeometry3D* pVolGeom,
	                      const CProjectionGeometry3D* pProjGeom)
{
	if (pData->initialized)
		return false;

	bool ok = convertAstraGeometry_dims(pVolGeom, pProjGeom, pData->dims);

	if (!ok)
		return false;

	pData->projs = 0;
	pData->parprojs = 0;

	ok = convertAstraGeometry(pVolGeom, pProjGeom,
	                          pData->parprojs, pData->projs,
	                          pData->params);
	if (!ok)
		return false;

	if (pData->projs) {
		assert(pData->parprojs == 0);
		pData->projType = PROJ_CONE;
	} else {
		assert(pData->parprojs != 0);
		pData->projType = PROJ_PARALLEL;
	}

	return true;
}


bool AstraSIRT3d::enableSuperSampling(unsigned int iVoxelSuperSampling,
                                      unsigned int iDetectorSuperSampling)
{
	if (pData->initialized)
		return false;

	if (iVoxelSuperSampling == 0 || iDetectorSuperSampling == 0)
		return false;

	pData->params.iRaysPerVoxelDim = iVoxelSuperSampling;
	pData->params.iRaysPerDetDim = iDetectorSuperSampling;

	return true;
}

bool AstraSIRT3d::setRelaxation(float r)
{
	if (pData->initialized)
		return false;

	pData->fRelaxation = r;

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
		ok = pData->sirt.setPar3DGeometry(pData->dims, pData->parprojs, pData->params);
	} else {
		ok = pData->sirt.setConeGeometry(pData->dims, pData->projs, pData->params);
	}

	if (!ok)
		return false;

	pData->sirt.setRelaxation(pData->fRelaxation);

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

float AstraSIRT3d::computeDiffNorm()
{
	if (!pData->initialized)
		return 0.0f; // FIXME: Error?

	return pData->sirt.computeDiffNorm();
}




class AstraCGLS3d_internal {
public:
	SDimensions3D dims;
	SProjectorParams3D params;
	CUDAProjectionType3d projType;

	float* angles;
	float fOriginSourceDistance;
	float fOriginDetectorDistance;

	SConeProjection* projs;
	SPar3DProjection* parprojs;

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

	pData->projs = 0;
	pData->parprojs = 0;

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

	delete[] pData->parprojs;
	pData->parprojs = 0;

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

bool AstraCGLS3d::setGeometry(const CVolumeGeometry3D* pVolGeom,
	                      const CProjectionGeometry3D* pProjGeom)
{
	if (pData->initialized)
		return false;

	bool ok = convertAstraGeometry_dims(pVolGeom, pProjGeom, pData->dims);

	if (!ok)
		return false;

	pData->projs = 0;
	pData->parprojs = 0;

	ok = convertAstraGeometry(pVolGeom, pProjGeom,
	                          pData->parprojs, pData->projs,
	                          pData->params);
	if (!ok)
		return false;

	if (pData->projs) {
		assert(pData->parprojs == 0);
		pData->projType = PROJ_CONE;
	} else {
		assert(pData->parprojs != 0);
		pData->projType = PROJ_PARALLEL;
	}

	return true;
}

bool AstraCGLS3d::enableSuperSampling(unsigned int iVoxelSuperSampling,
                                      unsigned int iDetectorSuperSampling)
{
	if (pData->initialized)
		return false;

	if (iVoxelSuperSampling == 0 || iDetectorSuperSampling == 0)
		return false;

	pData->params.iRaysPerVoxelDim = iVoxelSuperSampling;
	pData->params.iRaysPerDetDim = iDetectorSuperSampling;

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
		ok = pData->cgls.setPar3DGeometry(pData->dims, pData->parprojs, pData->params);
	} else {
		ok = pData->cgls.setConeGeometry(pData->dims, pData->projs, pData->params);
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

float AstraCGLS3d::computeDiffNorm()
{
	if (!pData->initialized)
		return 0.0f; // FIXME: Error?

	return pData->cgls.computeDiffNorm();
}



bool astraCudaFP(const float* pfVolume, float* pfProjections,
                 const CVolumeGeometry3D* pVolGeom,
                 const CProjectionGeometry3D* pProjGeom,
                 int iGPUIndex, int iDetectorSuperSampling,
                 Cuda3DProjectionKernel projKernel)
{
	SDimensions3D dims;
	SProjectorParams3D params;

	params.iRaysPerDetDim = iDetectorSuperSampling;

	bool ok = convertAstraGeometry_dims(pVolGeom, pProjGeom, dims);
	if (!ok)
		return false;

	if (iDetectorSuperSampling == 0)
		return false;

	SPar3DProjection* pParProjs;
	SConeProjection* pConeProjs;

	ok = convertAstraGeometry(pVolGeom, pProjGeom,
	                          pParProjs, pConeProjs,
	                          params);


	if (iGPUIndex != -1) {
		cudaSetDevice(iGPUIndex);
		cudaError_t err = cudaGetLastError();

		// Ignore errors caused by calling cudaSetDevice multiple times
		if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
			return false;
	}


	cudaPitchedPtr D_volumeData = allocateVolumeData(dims);
	ok = D_volumeData.ptr;
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

	if (pParProjs) {
		switch (projKernel) {
		case ker3d_default:
			ok &= Par3DFP(D_volumeData, D_projData, dims, pParProjs, params);
			break;
		case ker3d_sum_square_weights:
			ok &= Par3DFP_SumSqW(D_volumeData, D_projData, dims, pParProjs, params);
			break;
		default:
			assert(false);
		}
	} else {
		switch (projKernel) {
		case ker3d_default:
			ok &= ConeFP(D_volumeData, D_projData, dims, pConeProjs, params);
			break;
		default:
			assert(false);
		}
	}

	ok &= copyProjectionsFromDevice(pfProjections, D_projData,
	                                dims, dims.iProjU);


	cudaFree(D_volumeData.ptr);
	cudaFree(D_projData.ptr);

	return ok;

}


bool astraCudaBP(float* pfVolume, const float* pfProjections,
                 const CVolumeGeometry3D* pVolGeom,
                 const CProjectionGeometry3D* pProjGeom,
                 int iGPUIndex, int iVoxelSuperSampling)
{
	SDimensions3D dims;
	SProjectorParams3D params;

	params.iRaysPerVoxelDim = iVoxelSuperSampling;

	bool ok = convertAstraGeometry_dims(pVolGeom, pProjGeom, dims);
	if (!ok)
		return false;

	SPar3DProjection* pParProjs;
	SConeProjection* pConeProjs;

	ok = convertAstraGeometry(pVolGeom, pProjGeom,
	                          pParProjs, pConeProjs,
	                          params);

	if (iGPUIndex != -1) {
		cudaSetDevice(iGPUIndex);
		cudaError_t err = cudaGetLastError();

		// Ignore errors caused by calling cudaSetDevice multiple times
		if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess) {
			delete[] pParProjs;
			delete[] pConeProjs;
			return false;
		}
	}


	cudaPitchedPtr D_volumeData = allocateVolumeData(dims);
	ok = D_volumeData.ptr;
	if (!ok) {
		delete[] pParProjs;
		delete[] pConeProjs;
		return false;
	}

	cudaPitchedPtr D_projData = allocateProjectionData(dims);
	ok = D_projData.ptr;
	if (!ok) {
		delete[] pParProjs;
		delete[] pConeProjs;
		cudaFree(D_volumeData.ptr);
		return false;
	}

	ok &= copyProjectionsToDevice(pfProjections, D_projData,
	                              dims, dims.iProjU);

	ok &= zeroVolumeData(D_volumeData, dims);

	if (!ok) {
		delete[] pParProjs;
		delete[] pConeProjs;
		cudaFree(D_volumeData.ptr);
		cudaFree(D_projData.ptr);
		return false;
	}

	if (pParProjs)
		ok &= Par3DBP(D_volumeData, D_projData, dims, pParProjs, params);
	else
		ok &= ConeBP(D_volumeData, D_projData, dims, pConeProjs, params);

	ok &= copyVolumeFromDevice(pfVolume, D_volumeData, dims, dims.iVolX);

	delete[] pParProjs;
	delete[] pConeProjs;

	cudaFree(D_volumeData.ptr);
	cudaFree(D_projData.ptr);

	return ok;

}


// This computes the column weights, divides by them, and adds the
// result to the current volume. This is both more expensive and more
// GPU memory intensive than the regular BP, but allows saving system RAM.
bool astraCudaBP_SIRTWeighted(float* pfVolume,
                      const float* pfProjections,
                      const CVolumeGeometry3D* pVolGeom,
                      const CProjectionGeometry3D* pProjGeom,
                      int iGPUIndex, int iVoxelSuperSampling)
{
	SDimensions3D dims;
	SProjectorParams3D params;

	params.iRaysPerVoxelDim = iVoxelSuperSampling;

	bool ok = convertAstraGeometry_dims(pVolGeom, pProjGeom, dims);
	if (!ok)
		return false;


	SPar3DProjection* pParProjs;
	SConeProjection* pConeProjs;

	ok = convertAstraGeometry(pVolGeom, pProjGeom,
	                          pParProjs, pConeProjs,
	                          params);

	if (iGPUIndex != -1) {
		cudaSetDevice(iGPUIndex);
		cudaError_t err = cudaGetLastError();

		// Ignore errors caused by calling cudaSetDevice multiple times
		if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess) {
			delete[] pParProjs;
			delete[] pConeProjs;
			return false;
		}
	}


	cudaPitchedPtr D_pixelWeight = allocateVolumeData(dims);
	ok = D_pixelWeight.ptr;
	if (!ok) {
		delete[] pParProjs;
		delete[] pConeProjs;
		return false;
	}

	cudaPitchedPtr D_volumeData = allocateVolumeData(dims);
	ok = D_volumeData.ptr;
	if (!ok) {
		delete[] pParProjs;
		delete[] pConeProjs;
		cudaFree(D_pixelWeight.ptr);
		return false;
	}

	cudaPitchedPtr D_projData = allocateProjectionData(dims);
	ok = D_projData.ptr;
	if (!ok) {
		delete[] pParProjs;
		delete[] pConeProjs;
		cudaFree(D_pixelWeight.ptr);
		cudaFree(D_volumeData.ptr);
		return false;
	}

	// Compute weights
	ok &= zeroVolumeData(D_pixelWeight, dims);
	processSino3D<opSet>(D_projData, 1.0f, dims);

	if (pParProjs)
		ok &= Par3DBP(D_pixelWeight, D_projData, dims, pParProjs, params);
	else
		ok &= ConeBP(D_pixelWeight, D_projData, dims, pConeProjs, params);

	processVol3D<opInvert>(D_pixelWeight, dims);
	if (!ok) {
		delete[] pParProjs;
		delete[] pConeProjs;
		cudaFree(D_pixelWeight.ptr);
		cudaFree(D_volumeData.ptr);
		cudaFree(D_projData.ptr);
		return false;
	}

	ok &= copyProjectionsToDevice(pfProjections, D_projData,
	                              dims, dims.iProjU);
	ok &= zeroVolumeData(D_volumeData, dims);
	// Do BP into D_volumeData
	if (pParProjs)
		ok &= Par3DBP(D_volumeData, D_projData, dims, pParProjs, params);
	else
		ok &= ConeBP(D_volumeData, D_projData, dims, pConeProjs, params);

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

	delete[] pParProjs;
	delete[] pConeProjs;

	return ok;

}

_AstraExport bool uploadMultipleProjections(CFloat32ProjectionData3D *proj,
                                         const float *data,
                                         unsigned int y_min, unsigned int y_max)
{
	assert(proj->getStorage()->isGPU());
	CDataGPU *storage = dynamic_cast<CDataGPU*>(proj->getStorage());
	astraCUDA3d::MemHandle3D hnd = storage->getHandle();

	astraCUDA3d::SDimensions3D dims1;
	dims1.iProjU = proj->getDetectorColCount();
	dims1.iProjV = proj->getDetectorRowCount();
	dims1.iProjAngles = y_max - y_min + 1;

	cudaPitchedPtr D_proj = allocateProjectionData(dims1);
	bool ok = copyProjectionsToDevice(data, D_proj, dims1);
	if (!ok) {
		ASTRA_ERROR("Failed to upload projection to GPU");
		return false;
	}

	astraCUDA3d::MemHandle3D hnd1 = astraCUDA3d::wrapHandle(
			(float *)D_proj.ptr,
			dims1.iProjU, dims1.iProjAngles, dims1.iProjV,
			D_proj.pitch / sizeof(float));

	astraCUDA3d::SSubDimensions3D subdims;
	subdims.nx = dims1.iProjU;
	subdims.ny = proj->getAngleCount();
	subdims.nz = dims1.iProjV;
	subdims.pitch = D_proj.pitch / sizeof(float); // FIXME: Pitch for wrong obj!
	subdims.subnx = dims1.iProjU;
	subdims.subny = dims1.iProjAngles;
	subdims.subnz = dims1.iProjV;
	subdims.subx = 0;
	subdims.suby = y_min;
	subdims.subz = 0;

	ok = astraCUDA3d::copyIntoArray(hnd, hnd1, subdims);
	if (!ok) {
		ASTRA_ERROR("Failed to copy projection into 3d data");
		return false;
	}

	cudaFree(D_proj.ptr);
	return true;
}


}
