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

#include <memory>

namespace astra {
class CVolumeGeometry3D;
class CProjectionGeometry3D;
struct SFilterConfig;
class CData3D;
class CDataGPU;
}


// MemHandle3D defines a very basic opaque interface to GPU memory pointers.
// Its intended use is allowing ASTRA code to pass around GPU pointers without
// requiring CUDA headers.
//
// It generally wraps CUDA linear global memory.
//
// As a very basic extension, it also allows wrapping a CUDA 3D array.
// This extension (only) allows creating a CUDA 3D array, copying projection
// data into it, performing a BP from the array, and freeing the array.

namespace astraCUDA3d {

// TODO: Make it possible to delete these handles when they're no longer
// necessary inside the FP/BP
//
// TODO: Add functions for querying capacity

struct SMemHandle3D_internal;

struct MemHandle3D {
	std::shared_ptr<SMemHandle3D_internal> d;
	operator bool() const { return (bool)d; }
};

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

_AstraExport MemHandle3D wrapHandle(float *D_ptr, unsigned int x, unsigned int y, unsigned int z, unsigned int pitch);
MemHandle3D createProjectionArrayHandle(const float *ptr, unsigned int x, unsigned int y, unsigned int z);

astra::CDataGPU *allocateGPUMemory(unsigned int x, unsigned int y, unsigned int z, Mem3DZeroMode zero);

bool copyToGPUMemory(const astra::CData3D *src, astra::CData3D *dst, const SSubDimensions3D &pos);

bool copyFromGPUMemory(astra::CData3D *dst, const astra::CData3D *src, const SSubDimensions3D &pos);

bool freeGPUMemory(astra::CData3D *data);

bool zeroGPUMemory(astra::CData3D *data, unsigned int x, unsigned int y, unsigned int z);

bool setGPUIndex(int index);

bool copyIntoArray(MemHandle3D &handle, MemHandle3D &subdata, const SSubDimensions3D &pos);


bool FP(const astra::CProjectionGeometry3D* pProjGeom, astra::CData3D *projData, const astra::CVolumeGeometry3D* pVolGeom, astra::CData3D *volData, int iDetectorSuperSampling, astra::Cuda3DProjectionKernel projKernel);

bool BP(const astra::CProjectionGeometry3D* pProjGeom, astra::CData3D *projData, const astra::CVolumeGeometry3D* pVolGeom, astra::CData3D *volData, int iVoxelSuperSampling, astra::Cuda3DProjectionKernel projKernel);

bool FDK(const astra::CProjectionGeometry3D* pProjGeom, astra::CData3D *projData, const astra::CVolumeGeometry3D* pVolGeom, astra::CData3D *volData, bool bShortScan, const astra::SFilterConfig &filterConfig, float fOutputScale = 1.0f);

}

#endif
