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
#include "astra/cuda/2d/par_fp.h"
#include "astra/cuda/2d/fan_fp.h"
#include "astra/cuda/2d/par_bp.h"
#include "astra/cuda/2d/fan_bp.h"
#include "astra/cuda/2d/arith.h"
#include "astra/cuda/2d/astra.h"
#include "astra/cuda/2d/fft.h"

// For fan beam FBP weighting
#include "astra/cuda/3d/fdk.h"

#include "astra/GeometryUtil2D.h"
#include "astra/VolumeGeometry2D.h"
#include "astra/ParallelProjectionGeometry2D.h"
#include "astra/ParallelVecProjectionGeometry2D.h"
#include "astra/FanFlatProjectionGeometry2D.h"
#include "astra/FanFlatVecProjectionGeometry2D.h"
#include "astra/Logging.h"

#include <cstdio>
#include <cassert>
#include <fstream>

#include <cuda.h>

using namespace astraCUDA;
using namespace std;


namespace astra {

enum CUDAProjectionType {
	PROJ_PARALLEL,
	PROJ_FAN
};


BPalgo::BPalgo()
{

}

BPalgo::~BPalgo()
{

}

bool BPalgo::init()
{
	return true;
}

bool BPalgo::iterate(unsigned int)
{
	// TODO: This zeroVolume makes an earlier memcpy of D_volumeData redundant
	zeroVolumeData(D_volumeData, volumePitch, dims);
	callBP(D_volumeData, volumePitch, D_sinoData, sinoPitch, 1.0f);
	return true;
}

float BPalgo::computeDiffNorm()
{
	float *D_projData;
	unsigned int projPitch;

	allocateProjectionData(D_projData, projPitch, dims);

	duplicateProjectionData(D_projData, D_sinoData, sinoPitch, dims);
	callFP(D_volumeData, volumePitch, D_projData, projPitch, -1.0f);

	float s = dotProduct2D(D_projData, projPitch, dims.iProjDets, dims.iProjAngles);

	cudaFree(D_projData);

	return sqrt(s);
}


bool astraCudaFP(const float* pfVolume, float* pfSinogram,
                 unsigned int iVolWidth, unsigned int iVolHeight,
                 unsigned int iProjAngles, unsigned int iProjDets,
                 const SParProjection *pAngles,
                 unsigned int iDetSuperSampling,
                 float fOutputScale, int iGPUIndex)
{
	SDimensions dims;

	if (iProjAngles == 0 || iProjDets == 0 || pAngles == 0)
		return false;

	dims.iProjAngles = iProjAngles;
	dims.iProjDets = iProjDets;

	if (iDetSuperSampling == 0)
		return false;

	dims.iRaysPerDet = iDetSuperSampling;

	if (iVolWidth <= 0 || iVolHeight <= 0)
		return false;

	dims.iVolWidth = iVolWidth;
	dims.iVolHeight = iVolHeight;

	if (iGPUIndex != -1) {
		cudaSetDevice(iGPUIndex);
		cudaError_t err = cudaGetLastError();

		// Ignore errors caused by calling cudaSetDevice multiple times
		if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
			return false;
	}

	bool ok;

	float* D_volumeData;
	unsigned int volumePitch;

	ok = allocateVolumeData(D_volumeData, volumePitch, dims);
	if (!ok)
		return false;

	float* D_sinoData;
	unsigned int sinoPitch;

	ok = allocateProjectionData(D_sinoData, sinoPitch, dims);
	if (!ok) {
		cudaFree(D_volumeData);
		return false;
	}

	ok = copyVolumeToDevice(pfVolume, dims.iVolWidth,
	                        dims,
	                        D_volumeData, volumePitch);
	if (!ok) {
		cudaFree(D_volumeData);
		cudaFree(D_sinoData);
		return false;
	}

	zeroProjectionData(D_sinoData, sinoPitch, dims);
	ok = FP(D_volumeData, volumePitch, D_sinoData, sinoPitch, dims, pAngles, fOutputScale);
	if (!ok) {
		cudaFree(D_volumeData);
		cudaFree(D_sinoData);
		return false;
	}

	ok = copySinogramFromDevice(pfSinogram, dims.iProjDets,
	                            dims,
	                            D_sinoData, sinoPitch);
	if (!ok) {
		cudaFree(D_volumeData);
		cudaFree(D_sinoData);
		return false;
	}

	cudaFree(D_volumeData);
	cudaFree(D_sinoData);
	return true;
}

bool astraCudaFanFP(const float* pfVolume, float* pfSinogram,
                    unsigned int iVolWidth, unsigned int iVolHeight,
                    unsigned int iProjAngles, unsigned int iProjDets,
                    const SFanProjection *pAngles,
                    unsigned int iDetSuperSampling, float fOutputScale,
                    int iGPUIndex)
{
	SDimensions dims;

	if (iProjAngles == 0 || iProjDets == 0 || pAngles == 0)
		return false;

	dims.iProjAngles = iProjAngles;
	dims.iProjDets = iProjDets;

	if (iDetSuperSampling == 0)
		return false;

	dims.iRaysPerDet = iDetSuperSampling;

	if (iVolWidth <= 0 || iVolHeight <= 0)
		return false;

	dims.iVolWidth = iVolWidth;
	dims.iVolHeight = iVolHeight;

	if (iGPUIndex != -1) {
		cudaSetDevice(iGPUIndex);
		cudaError_t err = cudaGetLastError();

		// Ignore errors caused by calling cudaSetDevice multiple times
		if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
			return false;
	}

	bool ok;

	float* D_volumeData;
	unsigned int volumePitch;

	ok = allocateVolumeData(D_volumeData, volumePitch, dims);
	if (!ok)
		return false;

	float* D_sinoData;
	unsigned int sinoPitch;

	ok = allocateProjectionData(D_sinoData, sinoPitch, dims);
	if (!ok) {
		cudaFree(D_volumeData);
		return false;
	}

	ok = copyVolumeToDevice(pfVolume, dims.iVolWidth,
	                        dims,
	                        D_volumeData, volumePitch);
	if (!ok) {
		cudaFree(D_volumeData);
		cudaFree(D_sinoData);
		return false;
	}

	zeroProjectionData(D_sinoData, sinoPitch, dims);

	ok = FanFP(D_volumeData, volumePitch, D_sinoData, sinoPitch, dims, pAngles, fOutputScale);

	if (!ok) {
		cudaFree(D_volumeData);
		cudaFree(D_sinoData);
		return false;
	}

	ok = copySinogramFromDevice(pfSinogram, dims.iProjDets,
	                            dims,
	                            D_sinoData, sinoPitch);
	if (!ok) {
		cudaFree(D_volumeData);
		cudaFree(D_sinoData);
		return false;
	}

	cudaFree(D_volumeData);
	cudaFree(D_sinoData);

	return true;

}


// adjust pProjs to normalize volume geometry
template<typename ProjectionT>
static bool convertAstraGeometry_internal(const CVolumeGeometry2D* pVolGeom,
                          unsigned int iProjectionAngleCount,
                          ProjectionT*& pProjs,
                          float& fOutputScale)
{
	// TODO: Make EPS relative
	const float EPS = 0.00001f;

	// Check if pixels are square
	if (abs(pVolGeom->getPixelLengthX() - pVolGeom->getPixelLengthY()) > EPS)
		return false;

	float dx = -(pVolGeom->getWindowMinX() + pVolGeom->getWindowMaxX()) / 2;
	float dy = -(pVolGeom->getWindowMinY() + pVolGeom->getWindowMaxY()) / 2;

	float factor = 1.0f / pVolGeom->getPixelLengthX();

	for (int i = 0; i < iProjectionAngleCount; ++i) {
		// CHECKME: Order of scaling and translation
		pProjs[i].translate(dx, dy);
		pProjs[i].scale(factor);
	}
	// CHECKME: Check factor
	fOutputScale *= pVolGeom->getPixelLengthX() * pVolGeom->getPixelLengthY();

	return true;
}



bool convertAstraGeometry(const CVolumeGeometry2D* pVolGeom,
                          const CParallelProjectionGeometry2D* pProjGeom,
                          SParProjection*& pProjs,
                          float& fOutputScale)
{
	assert(pVolGeom);
	assert(pProjGeom);
	assert(pProjGeom->getProjectionAngles());

	int nth = pProjGeom->getProjectionAngleCount();

	pProjs = genParProjections(nth,
	                           pProjGeom->getDetectorCount(),
	                           pProjGeom->getDetectorWidth(),
	                           pProjGeom->getProjectionAngles(), 0);

	bool ok;
	fOutputScale = 1.0f;

	ok = convertAstraGeometry_internal(pVolGeom, nth, pProjs, fOutputScale);

	if (!ok) {
		delete[] pProjs;
		pProjs = 0;
	}

	return ok;
}

bool convertAstraGeometry(const CVolumeGeometry2D* pVolGeom,
                          const CParallelVecProjectionGeometry2D* pProjGeom,
                          SParProjection*& pProjs,
                          float& fOutputScale)
{
	assert(pVolGeom);
	assert(pProjGeom);
	assert(pProjGeom->getProjectionVectors());

	int nth = pProjGeom->getProjectionAngleCount();

	pProjs = new SParProjection[nth];

	for (int i = 0; i < nth; ++i) {
		pProjs[i] = pProjGeom->getProjectionVectors()[i];
	}

	bool ok;
	fOutputScale = 1.0f;

	ok = convertAstraGeometry_internal(pVolGeom, nth, pProjs, fOutputScale);

	if (!ok) {
		delete[] pProjs;
		pProjs = 0;
	}

	return ok;
}



bool convertAstraGeometry(const CVolumeGeometry2D* pVolGeom,
                          const CFanFlatProjectionGeometry2D* pProjGeom,
                          astraCUDA::SFanProjection*& pProjs,
                          float& outputScale)
{
	assert(pVolGeom);
	assert(pProjGeom);
	assert(pProjGeom->getProjectionAngles());

	// TODO: Make EPS relative
	const float EPS = 0.00001f;

	int nth = pProjGeom->getProjectionAngleCount();

	// Check if pixels are square
	if (abs(pVolGeom->getPixelLengthX() - pVolGeom->getPixelLengthY()) > EPS)
		return false;

	// TODO: Deprecate this.
//	if (pProjGeom->getExtraDetectorOffset())
//		return false;


	float fOriginSourceDistance = pProjGeom->getOriginSourceDistance();
	float fOriginDetectorDistance = pProjGeom->getOriginDetectorDistance();
	float fDetSize = pProjGeom->getDetectorWidth();
	const float *pfAngles = pProjGeom->getProjectionAngles();

	pProjs = genFanProjections(nth, pProjGeom->getDetectorCount(),
                               fOriginSourceDistance, fOriginDetectorDistance,
	                           fDetSize, pfAngles);

	convertAstraGeometry_internal(pVolGeom, nth, pProjs, outputScale);

	return true;

}

bool convertAstraGeometry(const CVolumeGeometry2D* pVolGeom,
                          const CFanFlatVecProjectionGeometry2D* pProjGeom,
                          astraCUDA::SFanProjection*& pProjs,
                          float& outputScale)
{
	assert(pVolGeom);
	assert(pProjGeom);
	assert(pProjGeom->getProjectionVectors());

	// TODO: Make EPS relative
	const float EPS = 0.00001f;

	int nx = pVolGeom->getGridColCount();
	int ny = pVolGeom->getGridRowCount();
	int nth = pProjGeom->getProjectionAngleCount();

	// Check if pixels are square
	if (abs(pVolGeom->getPixelLengthX() - pVolGeom->getPixelLengthY()) > EPS)
		return false;

	pProjs = new SFanProjection[nth];

	// Copy vectors
	for (int i = 0; i < nth; ++i)
		pProjs[i] = pProjGeom->getProjectionVectors()[i];

	convertAstraGeometry_internal(pVolGeom, nth, pProjs, outputScale);

	return true;
}

bool convertAstraGeometry(const CVolumeGeometry2D* pVolGeom,
                          const CProjectionGeometry2D* pProjGeom,
                          astraCUDA::SParProjection*& pParProjs,
                          astraCUDA::SFanProjection*& pFanProjs,
                          float& outputScale)
{
	const CParallelProjectionGeometry2D* parProjGeom = dynamic_cast<const CParallelProjectionGeometry2D*>(pProjGeom);
	const CParallelVecProjectionGeometry2D* parVecProjGeom = dynamic_cast<const CParallelVecProjectionGeometry2D*>(pProjGeom);
	const CFanFlatProjectionGeometry2D* fanProjGeom = dynamic_cast<const CFanFlatProjectionGeometry2D*>(pProjGeom);
	const CFanFlatVecProjectionGeometry2D* fanVecProjGeom = dynamic_cast<const CFanFlatVecProjectionGeometry2D*>(pProjGeom);

	bool ok = false;

	if (parProjGeom) {
		ok = convertAstraGeometry(pVolGeom, parProjGeom, pParProjs, outputScale);
	} else if (parVecProjGeom) {
		ok = convertAstraGeometry(pVolGeom, parVecProjGeom, pParProjs, outputScale);
	} else if (fanProjGeom) {
		ok = convertAstraGeometry(pVolGeom, fanProjGeom, pFanProjs, outputScale);
	} else if (fanVecProjGeom) {
		ok = convertAstraGeometry(pVolGeom, fanVecProjGeom, pFanProjs, outputScale);
	} else {
		ok = false;
	}

	return ok;
}

bool convertAstraGeometry_dims(const CVolumeGeometry2D* pVolGeom,
                               const CProjectionGeometry2D* pProjGeom,
                               SDimensions& dims)
{
	dims.iVolWidth = pVolGeom->getGridColCount();
	dims.iVolHeight = pVolGeom->getGridRowCount();

	dims.iProjAngles = pProjGeom->getProjectionAngleCount();
	dims.iProjDets = pProjGeom->getDetectorCount();

	dims.iRaysPerDet = 1;
	dims.iRaysPerPixelDim = 1;

	return true;
}



}

namespace astraCUDA {


_AstraExport std::string getCudaDeviceString(int device)
{
	char buf[1024];
	cudaError_t err;
	if (device == -1) {
		err = cudaGetDevice(&device);
		if (err != cudaSuccess) {
			return "Error getting current GPU index";
		}
	}

	cudaDeviceProp prop;
	err = cudaGetDeviceProperties(&prop, device);
	if (err != cudaSuccess) {
		snprintf(buf, 1024, "GPU #%d: Invalid device (%d): %s", device, err, cudaGetErrorString(err));
		return buf;
	}

	long mem = prop.totalGlobalMem / (1024*1024);
	snprintf(buf, 1024, "GPU #%d: %s, with %ldMB", device, prop.name, mem);
	return buf;
}

_AstraExport bool setGPUIndex(int iGPUIndex)
{
        if (iGPUIndex != -1) {
                cudaSetDevice(iGPUIndex);
                cudaError_t err = cudaGetLastError();

                // Ignore errors caused by calling cudaSetDevice multiple times
                if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
                        return false;
        }

        return true;
}

_AstraExport size_t availableGPUMemory()
{
    size_t free, total;
    cudaError_t err = cudaMemGetInfo(&free, &total);
    if (err != cudaSuccess)
        return 0;
    return free;
}




}
