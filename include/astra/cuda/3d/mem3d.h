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

#ifndef _CUDA_MEM3D_H
#define _CUDA_MEM3D_H

#include "astra3d.h"

namespace astra {
class CVolumeGeometry3D;
class CProjectionGeometry3D;
class CFloat32VolumeData3D;
class CFloat32ProjectionData3D;
struct SFilterConfig;
class CData3D;
class CDataStorage;
}


namespace astraCUDA3d {

struct SSubDimensions3D {
	unsigned int nx;
	unsigned int ny;
	unsigned int nz;
	unsigned int pitch;
	unsigned int subnx;
	unsigned int subny;
	unsigned int subnz;
	unsigned int subx;
	unsigned int suby;
	unsigned int subz;

	bool isFullVolume() const {
		return (subx == 0 && suby == 0 && subz == 0 &&
		        subnx == nx && subny == ny && subnz == nz);
	}
};

/*
// Useful or not?
enum Mem3DCopyMode {
	MODE_SET,
	MODE_ADD
};
*/

enum Mem3DZeroMode {
	INIT_NO,
	INIT_ZERO
};

int maxBlockDimension();

_AstraExport astra::CDataStorage* wrapHandle(float *D_ptr, unsigned int x, unsigned int y, unsigned int z, unsigned int pitch);
astra::CDataStorage* createProjectionArrayHandle(const float *ptr, unsigned int x, unsigned int y, unsigned int z);

astra::CDataStorage *allocateGPUMemory(unsigned int x, unsigned int y, unsigned int z, Mem3DZeroMode zero);

astra::CDataStorage *allocateGPUMemoryLike(const astra::CData3D *model, Mem3DZeroMode zero);
astra::CFloat32VolumeData3D *createGPUVolumeData3DLike(const astra::CFloat32VolumeData3D *model);
astra::CFloat32ProjectionData3D *createGPUProjectionData3DLike(const astra::CFloat32ProjectionData3D *model);

// Create base object without attached geometry
astra::CData3D *createGPUData3DLike(const astra::CData3D *model);

bool copyToGPUMemory(const astra::CData3D *src, astra::CData3D *dst);
bool copyToGPUMemory(const astra::CData3D *src, astra::CData3D *dst, const SSubDimensions3D &pos);

bool copyFromGPUMemory(astra::CData3D *dst, const astra::CData3D *src);
bool copyFromGPUMemory(astra::CData3D *dst, const astra::CData3D *src, const SSubDimensions3D &pos);


bool freeGPUMemory(astra::CData3D *data);

bool zeroGPUMemory(astra::CData3D *data);

bool assignGPUMemory(astra::CData3D *dst, const astra::CData3D *src);

bool copyIntoArray(astra::CData3D *data, astra::CData3D *subdata, const SSubDimensions3D &pos);


bool FP(astra::CFloat32ProjectionData3D *projData, const astra::CFloat32VolumeData3D *volData, int iDetectorSuperSampling, astra::Cuda3DProjectionKernel projKernel);
bool FP(astra::CData3D *projData, const astra::CData3D *volData, const astra::Geometry3DParameters &geom, SProjectorParams3D params);

bool BP(const astra::CFloat32ProjectionData3D *projData, astra::CFloat32VolumeData3D *volData, int iVoxelSuperSampling, astra::Cuda3DProjectionKernel projKernel);
bool BP(const astra::CData3D *projData, astra::CData3D *volData, const astra::Geometry3DParameters &geom, SProjectorParams3D params);

bool FDK(astra::CFloat32ProjectionData3D *projData, astra::CFloat32VolumeData3D *volData, bool bShortScan, const astra::SFilterConfig &filterConfig, float fOutputScale = 1.0f);

bool dotProduct3D(const astra::CData3D *D_data, float &fRet);

}

#endif
