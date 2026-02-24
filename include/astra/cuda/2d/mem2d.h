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

#ifndef _CUDA_MEM2D_H
#define _CUDA_MEM2D_H

// TODO: Remove this. Only for Mem3DZeroMode
#include "astra/cuda/3d/mem3d.h"

#include "astra/cuda/2d/astra.h"

namespace astra {

class CVolumeGeometry2D;
class CProjectionGeometry2D;
class CFloat32VolumeData2D;
class CFloat32ProjectionData2D;
class CData2D;
class CDataStorage;
}

namespace astraCUDA {

// TODO: Remove
using astraCUDA3d::Mem3DZeroMode;
using astraCUDA3d::INIT_NO;
using astraCUDA3d::INIT_ZERO;

_AstraExport astra::CDataStorage* wrapHandle(float *D_ptr, unsigned int x, unsigned int y, unsigned int pitch);

astra::CDataStorage *allocateGPUMemory(unsigned int x, unsigned int y, Mem3DZeroMode zero);

astra::CDataStorage *allocateGPUMemoryLike(const astra::CData2D *model, Mem3DZeroMode zero);
astra::CFloat32VolumeData2D *createGPUVolumeData2DLike(const astra::CFloat32VolumeData2D *model);
astra::CFloat32ProjectionData2D *createGPUProjectionData2DLike(const astra::CFloat32ProjectionData2D *model);

// Create base object without attached geometry
astra::CData2D *createGPUData2DLike(const astra::CData2D *model);

bool copyToGPUMemory(const astra::CData2D *src, astra::CData2D *dst);

bool copyFromGPUMemory(astra::CData2D *dst, const astra::CData2D *src);


bool freeGPUMemory(astra::CData2D *data);

bool zeroGPUMemory(astra::CData2D *data);

bool assignGPUMemory(astra::CData2D *dst, const astra::CData2D *src);

bool setGPUIndex(int index);

bool FP(astra::CFloat32ProjectionData2D *projData, const astra::CFloat32VolumeData2D *volData, int iDetectorSuperSampling, astra::Cuda2DProjectionKernel projKernel);
bool FP(astra::CData2D *projData, const astra::CData2D *volData, const astra::Geometry2DParameters &geom, SProjectorParams2D params);
bool FP_SART(astra::CData2D *projData, const astra::CData2D *volData, const astra::Geometry2DParameters &geom, SProjectorParams2D params, int angle);

bool BP(const astra::CFloat32ProjectionData2D *projData, astra::CFloat32VolumeData2D *volData, int iVoxelSuperSampling, astra::Cuda2DProjectionKernel projKernel);
bool BP(const astra::CData2D *projData, astra::CData2D *volData, const astra::Geometry2DParameters &geom, SProjectorParams2D params);
bool BP_SART(const astra::CData2D *projData, astra::CData2D *volData, const astra::Geometry2DParameters &geom, SProjectorParams2D params, int angle);

// TODO: This is currently defined in util, not in mem2d
float dotProduct2D(const astra::CData2D *D_data);

}



#endif
