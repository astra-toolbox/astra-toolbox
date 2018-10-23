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

#ifndef _CUDA_MEM3D_H
#define _CUDA_MEM3D_H

#include <boost/shared_ptr.hpp>

#include "astra3d.h"

namespace astra {
class CVolumeGeometry3D;
class CProjectionGeometry3D;	
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
	boost::shared_ptr<SMemHandle3D_internal> d;
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

MemHandle3D allocateGPUMemory(unsigned int x, unsigned int y, unsigned int z, Mem3DZeroMode zero);

bool copyToGPUMemory(const float *src, MemHandle3D dst, const SSubDimensions3D &pos);

bool copyFromGPUMemory(float *dst, MemHandle3D src, const SSubDimensions3D &pos);

bool freeGPUMemory(MemHandle3D handle);

bool zeroGPUMemory(MemHandle3D handle, unsigned int x, unsigned int y, unsigned int z);

bool setGPUIndex(int index);

bool copyIntoArray(MemHandle3D handle, MemHandle3D subdata, const SSubDimensions3D &pos);


bool FP(const astra::CProjectionGeometry3D* pProjGeom, MemHandle3D projData, const astra::CVolumeGeometry3D* pVolGeom, MemHandle3D volData, int iDetectorSuperSampling, astra::Cuda3DProjectionKernel projKernel);

bool BP(const astra::CProjectionGeometry3D* pProjGeom, MemHandle3D projData, const astra::CVolumeGeometry3D* pVolGeom, MemHandle3D volData, int iVoxelSuperSampling, bool bFDKWeighting);

bool FDK(const astra::CProjectionGeometry3D* pProjGeom, MemHandle3D projData, const astra::CVolumeGeometry3D* pVolGeom, MemHandle3D volData, bool bShortScan, const float *pfFilter = 0);

}

#endif
