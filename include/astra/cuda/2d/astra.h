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

#ifndef _CUDA_ASTRA_H
#define _CUDA_ASTRA_H

#include "dims.h"
#include "algo.h"

using astraCUDA::SFanProjection;

namespace astra {

enum Cuda2DProjectionKernel {
	ker2d_default = 0
};

class CParallelProjectionGeometry2D;
class CParallelVecProjectionGeometry2D;
class CFanFlatProjectionGeometry2D;
class CFanFlatVecProjectionGeometry2D;
class CVolumeGeometry2D;
class CProjectionGeometry2D;


class _AstraExport BPalgo : public astraCUDA::ReconAlgo {
public:
	BPalgo();
	~BPalgo();

	virtual bool init();

	virtual bool iterate(unsigned int iterations);

	virtual float computeDiffNorm();
};




// TODO: Clean up this interface to FP

// Do a single forward projection
_AstraExport bool astraCudaFP(const float* pfVolume, float* pfSinogram,
                 unsigned int iVolWidth, unsigned int iVolHeight,
                 unsigned int iProjAngles, unsigned int iProjDets,
                 const SParProjection *pAngles,
                 unsigned int iDetSuperSampling = 1,
                 float fOutputScale = 1.0f, int iGPUIndex = 0);

_AstraExport bool astraCudaFanFP(const float* pfVolume, float* pfSinogram,
                    unsigned int iVolWidth, unsigned int iVolHeight,
                    unsigned int iProjAngles, unsigned int iProjDets,
                    const SFanProjection *pAngles,
                    unsigned int iDetSuperSampling = 1,
                    float fOutputScale = 1.0f, int iGPUIndex = 0);


_AstraExport bool convertAstraGeometry(const CVolumeGeometry2D* pVolGeom,
                    const CParallelProjectionGeometry2D* pProjGeom,
                    astraCUDA::SParProjection*& pProjs,
                    float& fOutputScale);

_AstraExport bool convertAstraGeometry(const CVolumeGeometry2D* pVolGeom,
                    const CParallelVecProjectionGeometry2D* pProjGeom,
                    astraCUDA::SParProjection*& pProjs,
                    float& fOutputScale);


_AstraExport bool convertAstraGeometry(const CVolumeGeometry2D* pVolGeom,
                    const CFanFlatProjectionGeometry2D* pProjGeom,
                    astraCUDA::SFanProjection*& pProjs,
                    float& outputScale);

_AstraExport bool convertAstraGeometry(const CVolumeGeometry2D* pVolGeom,
                    const CFanFlatVecProjectionGeometry2D* pProjGeom,
                    astraCUDA::SFanProjection*& pProjs,
                    float& outputScale);

_AstraExport bool convertAstraGeometry_dims(const CVolumeGeometry2D* pVolGeom,
                          const CProjectionGeometry2D* pProjGeom,
                          astraCUDA::SDimensions& dims);

_AstraExport bool convertAstraGeometry(const CVolumeGeometry2D* pVolGeom,
                          const CProjectionGeometry2D* pProjGeom,
                          astraCUDA::SParProjection*& pParProjs,
                          astraCUDA::SFanProjection*& pFanProjs,
                          float& outputScale);
}

namespace astraCUDA {

// Return string with CUDA device number, name and memory size.
// Use device == -1 to get info for the current device.
_AstraExport std::string getCudaDeviceString(int device);

_AstraExport bool setGPUIndex(int index);

_AstraExport size_t availableGPUMemory();

}
#endif
